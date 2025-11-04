"""
train_omnivla.py

Train or finetune OmniVLA with LoRA.
"""

# ==============================
# Configuration Flags
# ==============================
TRAIN_MODE = False   # True: training mode, False: debug mode (minimize GPU RAM usage)
VISUALIZE = True    # True: save visualization images of policy performance

# ==============================
# Path Setup
# ==============================
import sys
from pathlib import Path

# Add external project paths if not installed as packages
sys.path.extend([
    "../Learning-to-Drive-Anywhere-with-MBRA/train/",
])

# ==============================
# Standard Libraries
# ==============================
import os
import time
import math
import json
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from PIL import Image
from collections import deque, OrderedDict
from typing import Dict, Optional, Tuple, Type
from dataclasses import dataclass
from pathlib import Path
from torchvision import transforms

# ==============================
# Environment Settings
# ==============================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "60"
os.environ["MKL_NUM_THREADS"] = "60"
torch.set_num_threads(60)

# ==============================
# Third-Party Libraries
# ==============================
import tqdm
import wandb
import draccus
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# ==============================
# OmniVLA & Prismatic Modules
# ==============================
from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask

from prismatic.util.data_utils import PaddedCollatorForActionPrediction_Nav_MMN
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM, IGNORE_INDEX
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.dummy_dataset import Dummy_Dataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from vint_train.models.exaug.exaug import ExAug_dist_delay

# ==============================
# Transform Definition
# ==============================
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@dataclass
class OmniVLAConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)

    # Training configuration
    batch_size: int = 1                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 1e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    save_freq: int = 1000                          # Checkpoint saving frequency in steps    
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!
    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

def remove_ddp_in_checkpoint(state_dict) -> dict:
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def get_run_id(cfg) -> str:
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.vla_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id

def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt")) and module_name == "pose_projector":
        module_name = "proprio_projector"
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)

def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)

def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")

def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: OmniVLAConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:

    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
        
    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)

def sinc_apx(angle):
    return torch.sin(3.141592*angle + 0.000000001)/(3.141592*angle + 0.000000001)

def twist_to_pose_diff_torch(v, w, dt):
    theta = -w  * dt
    z = v * dt * sinc_apx(-theta / np.pi)
    x = -v * dt * sinc_apx(-theta / (2 * np.pi)) * torch.sin(-theta / 2)
    return x, z, theta

def robot_pos_model(linear_vel, angular_vel):
    # velocity commands integral
    bs, chorizon = linear_vel.shape
    device = linear_vel.device

    px = []
    pz = []
    pyaw = []
    Tacc = torch.eye(4, 4).unsqueeze(0).repeat(bs,1,1).to(device)
    for i in range(chorizon):
        x, z, yaw = twist_to_pose_diff_torch(linear_vel[:, i], angular_vel[:, i], 0.333)
        Todom = torch.zeros((bs, 4, 4)).to(device)
        Todom[:, 0, 0] = torch.cos(yaw)
        Todom[:, 0, 2] = torch.sin(yaw)
        Todom[:, 1, 1] = 1.0
        Todom[:, 2, 0] = -torch.sin(yaw)
        Todom[:, 2, 2] = torch.cos(yaw)
        Todom[:, 0, 3] = x
        Todom[:, 2, 3] = z
        Todom[:, 3, 3] = 1.0        
        
        Tacc = torch.matmul(Tacc, Todom)
               
        pyaw.append(torch.arctan(Tacc[:, 0, 2]/(Tacc[:, 0, 0] + 0.000000001)))        
        px.append(Tacc[:, 0, 3])
        pz.append(Tacc[:, 2, 3])   
    
    px_ref_list = px
    pz_ref_list = pz
    ry_ref_list = pyaw
    
    x_traj = []
    z_traj = []
    yaw_traj = [] 
    for ic in range(len(px_ref_list)):
        x_traj.append(px_ref_list[ic].unsqueeze(1))
        z_traj.append(pz_ref_list[ic].unsqueeze(1))
        yaw_traj.append(ry_ref_list[ic].unsqueeze(1))                            
    x_traj_cat = torch.cat(x_traj, axis = 1)
    z_traj_cat = torch.cat(z_traj, axis = 1)
    yaw_traj_cat = torch.cat(yaw_traj, axis = 1)                        
            
    metric_waypoint_spacing = 0.25*0.5
    # camera coordinate --> robot coordinate 
    action_estfrod = torch.cat((z_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, -x_traj_cat.unsqueeze(-1)/metric_waypoint_spacing, torch.cos(-yaw_traj_cat).unsqueeze(-1), torch.sin(-yaw_traj_cat).unsqueeze(-1)), axis=2)         
             
    return action_estfrod    

def run_forward_pass(
    vla,
    action_head,
    mbra,
    pose_projector,
    batch,
    action_tokenizer,
    device_id,
    num_patches,
    idrun=0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction_MMN): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        mbra: MBRA action reannotater
        pose_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        num_patches (int): Number of vision patches.
        idrun: iteration number

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}
    context_size = 5

    #batch size
    Bsize = batch["cur_image"].size()[0]
    
    # Get ground-truth action labels    
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    modality_id = batch["goal_mask_select"]
        
    #MBRA action reannotation
    img_goal = transform(batch["goal_image_8"])
    img_hist = torch.split(batch["cur_image"], 3, dim=1)
    img_hist_norm_list = [transform(obs_image) for obs_image in img_hist]
    img_hist_norm = torch.concat(img_hist_norm_list, dim=1)      

    rsize = 0.3*torch.ones(Bsize, 1, 1).to(device_id)
    delay = torch.zeros(Bsize, 1, 1).to(device_id)
    linear_vel_old = 0.5*torch.ones(Bsize, 6).float().to(device_id)
    angular_vel_old = 0.0*torch.ones(Bsize, 6).float().to(device_id)
    vel_past = torch.cat((linear_vel_old, angular_vel_old), axis=1).unsqueeze(2)          
                
    with torch.no_grad():
        linear_vel, angular_vel, _ = mbra(img_hist_norm, img_goal, rsize, delay, vel_past)  
    action_mbra = robot_pos_model(linear_vel, angular_vel)  
    
    # OmniVLA forward pass    
    if TRAIN_MODE:
        with torch.autocast("cuda", dtype=torch.bfloat16):    
            output: CausalLMOutputWithPast = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                attention_mask_label=batch["attention_mask_label"].to(device_id),                  
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                modality_id=modality_id.to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id),                
                proprio_projector=pose_projector,
                use_film=False,
            )
    else:
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    attention_mask_label=batch["attention_mask_label"].to(device_id),                    
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),                   
                    modality_id=modality_id.to(torch.bfloat16).to(device_id),                                       
                    labels=batch["labels"],
                    output_hidden_states=True,                   
                    proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id),
                    proprio_projector=pose_projector,
                    use_film=False,
                )
    
    # Get object pose
    obj_pose_norm = batch["obj_pose_norm"].to(dtype=torch.bfloat16).to(device_id)       
    # Get action masks needed for logging
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Compute metrics for continuous action representations (L1 regression | diffusion)
    if True:
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)

        # Predict action
        if TRAIN_MODE:
            predicted_actions = action_head.module.predict_action(actions_hidden_states, modality_id.to(torch.bfloat16).to(device_id))
        else:
            with torch.no_grad():
                predicted_actions = action_head.module.predict_action(actions_hidden_states, modality_id.to(torch.bfloat16).to(device_id))
                                            
        # Setting supervised action command by raw action or systhetic MBRA action
        mask_act = batch["action_select_mask"].to(torch.bfloat16).to(device_id).unsqueeze(1).unsqueeze(2).repeat(1,8,4)
        mask_notact = -1.0*(mask_act - 1.0)           
        action_ref = mask_act*ground_truth_actions + mask_notact*action_mbra.detach().to(torch.bfloat16)

        limited_temp_dist = torch.clip(batch["temp_dist"], min=0.0, max=20.0) 
        lan_bool = (batch["goal_mask_select"] == 7)|(batch["goal_mask_select"] == 8) #object loss is only for the LeLaN dataset
        loss = 1.0*torch.nn.MSELoss()(action_ref, predicted_actions) + 0.1*torch.nn.MSELoss()(obj_pose_norm[lan_bool], predicted_actions[:,-1,0:2][lan_bool]) + 0.1*torch.nn.MSELoss()(predicted_actions[:,0:-1], predicted_actions[:,1:])            
        L2_action = torch.nn.MSELoss()(action_ref, predicted_actions)
        L2_obj = torch.nn.MSELoss()(obj_pose_norm[lan_bool], predicted_actions[:,-1,0:2][lan_bool])
        L2_smooth = torch.nn.MSELoss()(predicted_actions[:,0:-1], predicted_actions[:,1:])
            
        loss_list = []
        task_list = []
        for icl in range(9):
            mask_task = batch["goal_mask_select"] == icl
            L2_action_task = torch.nn.MSELoss()(action_ref[mask_task], predicted_actions[mask_task])
            loss_list.append(L2_action_task)
            task_list.append(torch.sum(mask_task.float()))

        metrics.update(
            {
                "loss_value": loss.item(),            # Detached value for logging
                "L2_action_value": L2_action.item(),  # Detached value for logging                
                "L2_obj_value": L2_obj.item(),        # Detached value for logging
                "L2_smooth_value": L2_smooth.item(),  # Detached value for logging                  
                "L2_sate": loss_list[0].item(),
                "L2_sate_pose": loss_list[1].item(),
                "L2_sate_img": loss_list[2].item(),  
                "L2_sate_pose_img": loss_list[3].item(),                                                                           
                "L2_pose": loss_list[4].item(),
                "L2_pose_img": loss_list[5].item(),
                "L2_img": loss_list[6].item(),  
                "L2_lan": loss_list[7].item(),         
                "L2_lan_pose": loss_list[8].item(),                                
            }
        )

        if VISUALIZE == True:
            visualize_train(
                batch["img_PIL"],
                batch["gimg_PIL"],              
                obj_pose_norm.detach().cpu(),   
                batch["goal_pose"].detach().cpu(),
                ground_truth_actions.detach().cpu(),
                action_mbra.detach().cpu(),    
                predicted_actions.detach().cpu(),   
                action_ref.detach().cpu(),    
                batch["goal_mask_select"],          
                "train",   
                0,         
                idrun,                              
                1,                  
                False,                               
                )                                        

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss, metrics

def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Compute smoothened metrics from recent deques.

    Args:
        metrics_deques (dict): Dictionary of deques containing recent metrics.

    Returns:
        dict: Dictionary of smoothened metrics.
    """
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            #smoothened_metrics[name] = sum(deque) / len(deque)
            valid_values = [x for x in deque if not math.isnan(x)]
            if len(valid_values) == 0:
                smoothened_metrics[name] = math.nan
            else:
                smoothened_metrics[name] = sum(valid_values) / len(valid_values)
            
    return smoothened_metrics

def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Log metrics to Weights & Biases.

    Args:
        metrics (dict): Dictionary of metrics to log
        prefix (str): Prefix for metric names
        step (int): Training step
        wandb_entity (str): W&B entity instance

    Returns:
        None.
    """
    log_dict = {}
    for name, value in metrics.items():
        # Map loss_value to Loss for better readability in W&B
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        # Keep other metrics as is
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    pose_projector,
    action_head,
    distributed_state,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (OmniVLAConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction_MMN): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        pose_projector (nn.Module): Proprioceptive state projector module.
        action_head (nn.Module): Action head module.
        distributed_state (PartialState): Distributed training state.

    Returns:
        None.
    """
    # Determine checkpoint paths and naming
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    # Create directories and save dataset statistics (main process only)
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)

        # Save other components
        torch.save(pose_projector.state_dict(), checkpoint_dir / f"pose_projector--{checkpoint_name_suffix}")
        torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

    # Wait for model components to be saved
    dist.barrier()

    # Merge LoRA weights into base model and save resulting model checkpoint
    # Note: Can be very slow on some devices; if so, we recommend merging offline
    if cfg.use_lora and cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )#
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        # Wait for merged model to be saved
        dist.barrier()

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float32).numpy()

def visualize_train(
    batch_current_PIL: torch.Tensor,
    batch_goal_PIL: torch.Tensor,  
    goal_pos_lan: torch.Tensor, 
    goal_pos: torch.Tensor, 
    traj_raw: torch.Tensor,
    traj_mbra: torch.Tensor,    
    est_traj: torch.Tensor,
    select_traj: torch.Tensor,    
    goal_mask_select: torch.Tensor,
    eval_type: str,    
    epoch: int,
    count: int,
    num_images_log: int = 10,            
    lan: bool = True,    
):
    """Plot samples from the exploration model."""
    project_folder = "/raid/users/noriaki/openvla-oft/visualization"
    project_folder = "./visualization"
    visualize_path = os.path.join(
        project_folder,
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )        
    
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)
        
    wandb_list = []
    
    if lan:
        goal_pos_gt = goal_pos_lan #object pose (Language conditioned nav on LeLaN dataset only)
    else:
        goal_pos_gt = goal_pos #goal pose
    
    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,2)
        ax_graph = fig.add_subplot(gs[0:2, 1:2])      
        ax_ob = fig.add_subplot(gs[0:1, 0:1])
        ax_goal = fig.add_subplot(gs[1:2, 0:1])   

        ax_ob.imshow(np.array(batch_current_PIL[i]).astype(np.uint8))
        ax_goal.imshow(np.array(batch_goal_PIL[i]).astype(np.uint8))                  
                                            
        xgt = to_numpy(goal_pos_gt[i,0])
        ygt = to_numpy(goal_pos_gt[i,1])
        task_id = goal_mask_select[i].item()
            
        x_raw = traj_raw[i, :, 0].detach().cpu().to(torch.float32).numpy()
        y_raw = traj_raw[i, :, 1].detach().cpu().to(torch.float32).numpy()
        x_mbra = traj_mbra[i, :, 0].detach().cpu().to(torch.float32).numpy()
        y_mbra = traj_mbra[i, :, 1].detach().cpu().to(torch.float32).numpy()          
        x_est = est_traj[i, :, 0].detach().cpu().to(torch.float32).numpy()
        y_est = est_traj[i, :, 1].detach().cpu().to(torch.float32).numpy()          
        x_select = select_traj[i, :, 0].detach().cpu().to(torch.float32).numpy()
        y_select = select_traj[i, :, 1].detach().cpu().to(torch.float32).numpy()

        ax_graph.plot(-y_select, x_select, marker = 'o', color='m', linewidth=4, markersize=10, label="select") 
        ax_graph.plot(-np.insert(y_est, 0, 0.0), np.insert(x_est, 0, 0.0), linewidth=4.0, markersize=12, marker='o', color='blue', label="raw")                                                      
        ax_graph.plot(-y_raw, x_raw, marker = 'o', color='red', label="raw")
        ax_graph.plot(-y_mbra, x_mbra, marker = 'o', color='green', label="mbra")                                                
        ax_graph.plot(-ygt, xgt, marker = '*', color='red')   
        ax_graph.text(2.5, -0.2, str(task_id))

        mask_type = int(task_id)
        mask_texts = [
            "satellite only", "pose and satellite", "satellite and image", "all",
            "pose only", "pose and image", "image only", "language only", "language and pose"
        ]
        if mask_type < len(mask_texts):
            ax_graph.annotate(mask_texts[mask_type], xy=(1.0, 0.0), xytext=(-20, 20), fontsize=18, textcoords='offset points')
                                                   
        # set title
        ax_graph.set_title(f"est. trajectory (normzlied dim.)")
        ax_graph.set_xlim(-3.0, 3.0)
        ax_graph.set_ylim(-0.1, 10.0)
        ax_graph.legend(loc='best')                  
        ax_ob.set_title("Egocentric current image", fontsize=18)
        ax_goal.set_title("Egocentric goal image", fontsize=18)                     
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_{count}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))        
        plt.close(fig)

def merge_batches_padding(batch_list, pad_token_id, IGNORE_INDEX, model_max_length):
    """
    Merge a list of dictionary batches into a single dictionary,
    concatenating tensor values along the batch dimension (dim=0).
    """
    merged = {}
    keys = batch_list[0].keys()
    for key in keys:
        values = [batch[key] for batch in batch_list]
        first_value = values[0]

        if isinstance(first_value, torch.Tensor):
            merged[key] = torch.cat(values, dim=0)
        elif isinstance(first_value, list):
            combined_list = []
            for v in values:
                combined_list.extend(v)
            merged[key] = combined_list            
        else:
            pass  # or merged[key] = batch_list[0][key]

    input_ids = pad_sequence(merged["input_ids"], batch_first=True, padding_value=pad_token_id)
    merged["input_ids"] = input_ids[:, : model_max_length]
    labels = pad_sequence(merged["labels"], batch_first=True, padding_value=IGNORE_INDEX)
    merged["labels"] = labels[:, : model_max_length]
    merged["attention_mask"] = merged["input_ids"].ne(pad_token_id)        
    merged["attention_mask_label"] = merged["labels"].ne(IGNORE_INDEX)            
    merged["goal_mask_select"] = torch.tensor(merged["modality_id"])
    return merged

@draccus.wrap()
def train_omnivla(cfg: OmniVLAConfig) -> None:
    """
    Training OmniVLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (OmniVLAConfig): Training configuration.

    Returns:
        None.
    """
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    if cfg.vla_path == "openvla/openvla-7b": #from OpenVLA checkpoints
        cfg.resume = False
        cfg.resume_step = None
    elif cfg.vla_path == "./omnivla-original": #from OmniVLA checkpoints (paper version)
        cfg.resume = True     
        cfg.resume_step = 120000 
    elif cfg.vla_path == "./omnivla-original-balance": #from OmniVLA checkpoints (fix LeLaN data unbalance)
        cfg.resume = True     
        cfg.resume_step = 285000  
    elif cfg.vla_path == "./omnivla-finetuned-cast": #from OmniVLA checkpoints fituned with CAST dataset 
        cfg.resume = True      
        cfg.resume_step = 210000 
                                                                                          
    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)
    print("run_dir", run_dir, run_id)
        
    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    world_size = int(os.environ["WORLD_SIZE"]) 
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    print("World size", world_size, "rank", device_id)

    # Initialize wandb logging
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")
    
    #defining and loading MBRA
    with open("./config_nav/mbra_config.yaml", "r") as f:        
        config = yaml.safe_load(f)      
    mbra = ExAug_dist_delay(
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        obs_encoder=config["obs_encoder"],
        obs_encoding_size=config["obs_encoding_size"],
        late_fusion=config["late_fusion"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )     
    checkpoint_path_mbra = os.path.join("./MBRA", "mbra.pth")
    print("Loading MBRA model from ", checkpoint_path_mbra)
    latest_checkpoint_mbra = torch.load(checkpoint_path_mbra, map_location="cpu")
    mbra.load_state_dict(latest_checkpoint_mbra, strict=False) 
    mbra.eval().to(device=device_id)
    mbra = wrap_ddp(mbra, device_id, find_unused=True)

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPOSE_DIM: {POSE_DIM}\n"
    )

    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect
    print("model_is_on_hf_hub(cfg.vla_path)", model_is_on_hf_hub(cfg.vla_path))
    Load_hf = model_is_on_hf_hub(cfg.vla_path)
    if Load_hf:
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        # Overwrite VLA path
        cfg.vla_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)
    # Wait for model files to be synced
    dist.barrier()

    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True) #
    
    if Load_hf:
        index_file =  cfg.vla_path + "/model.safetensors.index.json"
        with open(index_file, "r") as f:
            index = json.load(f)

        # Extract unique filenames (strings)
        filenames = set(index["weight_map"].values())
    
        from safetensors.torch import load_file
        state_dict = {}
        for fname in filenames:
            shard_path = os.path.join(cfg.vla_path, fname)
            shard_state = load_file(shard_path)
            state_dict.update(shard_state)    

        config_openvla = AutoConfig.from_pretrained(cfg.vla_path, trust_remote_code=True)        #
        vla = OpenVLAForActionPrediction_MMNv1(config_openvla)
        vla.load_state_dict(state_dict, strict=False)
    
    else:
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device_id) #            trust_remote_code=True,
    
    print("vla class", type(vla))
    print("llm class", type(vla.language_model))

    # Set number of images in VLA input
    print("cfg.num_images_in_input", cfg.num_images_in_input)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.to(dtype=torch.bfloat16, device=device_id)

    # LoRA setup
    target_modules = []
    
    for name, module in vla.named_modules():
        if isinstance(module, torch.nn.Linear):
            target_modules.append(name)    
    
    
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector
    pose_projector = init_module(
        ProprioProjector,
        "pose_projector",
        cfg,
        device_id,
        {"llm_dim": vla.module.llm_dim, "proprio_dim": POSE_DIM},
    )

    # If applicable, instantiate continuous action head for L1 regression
    action_head = init_module(
        L1RegressionActionHead_idcat,
        "action_head",
        cfg,
        device_id,
        {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
        to_bf16=True,
    )

    # Get number of vision patches
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # For goal pose conditioning
    NUM_PATCHES += 1

    if not TRAIN_MODE:
        for param in vla.parameters():
            param.requires_grad = False
                
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    trainable_params += [param for param in pose_projector.parameters() if param.requires_grad]
    print(f"# total trainable params: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Record original learning rate
    original_lr = optimizer.param_groups[0]["lr"]

    # Create learning rate scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],  # Number of steps after which LR will change
        gamma=0.1,  # Multiplicative factor of learning rate decay
    )

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1

    # Create collator and dataloader
    tokenizer_max_length = processor.tokenizer.model_max_length
    collator = PaddedCollatorForActionPrediction_Nav_MMN(
        tokenizer_max_length, processor.tokenizer.pad_token_id, padding_side="right", num_img = cfg.num_images_in_input
        
    )

    #Data loader and sampler setting (I provide the sample dataloader. Please replace this dataloader with your dataset. Following sample code, you can combine the mutiple datasets.)        
    train_dataset_dummy = []
    test_dataset_dummy = []    
    for data_split_type in ["train", "test"]:   
        #dummy dataset
        dataset_dummy = Dummy_Dataset(   
            context_size = config["context_size"],             
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer, 
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,                                                                         
        ) 
        if data_split_type == "train":
            train_dataset_dummy.append(dataset_dummy)
        elif data_split_type == "test":
            test_dataset_dummy.append(dataset_dummy)
                    
        if data_split_type == "train":                   
            train_dataset_dummy = ConcatDataset(train_dataset_dummy)
            sampler_train_dummy = DistributedSampler(train_dataset_dummy, num_replicas=world_size, rank=device_id, shuffle=True) 
                
            train_loader_dummy = DataLoader(
                train_dataset_dummy,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=8,
                drop_last=True,
                persistent_workers=True,
                sampler=sampler_train_dummy,
            )                  
        else:
            test_dataset_dummy = ConcatDataset(test_dataset_dummy) 
            sampler_test_dummy = DistributedSampler(test_dataset_dummy, num_replicas=world_size, rank=device_id, shuffle=True)                 

            test_loader_dummy = DataLoader(
                test_dataset_dummy,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=8,
                drop_last=True,
                persistent_workers=True,
                sampler=sampler_train_dummy,
            )

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "L2_action_value": deque(maxlen=cfg.grad_accumulation_steps),        
        "L2_obj_value": deque(maxlen=cfg.grad_accumulation_steps),
        "L2_smooth_value": deque(maxlen=cfg.grad_accumulation_steps),             
        "L2_sate": deque(maxlen=cfg.grad_accumulation_steps),
        "L2_sate_pose": deque(maxlen=cfg.grad_accumulation_steps),        
        "L2_sate_img": deque(maxlen=cfg.grad_accumulation_steps),
        "L2_sate_pose_img": deque(maxlen=cfg.grad_accumulation_steps),   
        "L2_pose": deque(maxlen=cfg.grad_accumulation_steps),        
        "L2_pose_img": deque(maxlen=cfg.grad_accumulation_steps),
        "L2_img": deque(maxlen=cfg.grad_accumulation_steps),       
        "L2_lan": deque(maxlen=cfg.grad_accumulation_steps),          
        "L2_lan_pose": deque(maxlen=cfg.grad_accumulation_steps),                                            
    }

    #You can list your all training datasets. In this example, we list same two dummy data loaders.
    iters = [iter(train_loader_dummy), iter(train_loader_dummy)]
    samplers = [sampler_train_dummy, sampler_train_dummy]     
                 
    log_count = 0
    for epoch in range(100):
        for sampler in samplers:
            sampler.set_epoch(epoch)
                
        with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
            if TRAIN_MODE:
                print("setting up training mode")
                vla.train()
            else:
                print("setting up eval (Local PC coding) mode")
                vla.eval()
                action_head.eval()
                pose_projector.eval()
                
            optimizer.zero_grad()
            for batch_idx in range(cfg.max_steps):
                batches = []
                for i, it in enumerate(iters):
                    try:
                        batch = next(it)
                    except StopIteration:
                        iters[i] = iter([iter(train_loader_dummy), iter(train_loader_dummy)][i])
                        batch = next(iters[i])
                    batches.append(batch)
                
                #Merging multiple datasets
                merged_batch = merge_batches_padding(batches, processor.tokenizer.pad_token_id, IGNORE_INDEX, tokenizer_max_length)                  

                # Compute training metrics and loss
                loss, metrics = run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    mbra=mbra,
                    pose_projector=pose_projector,
                    batch=merged_batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    num_patches=NUM_PATCHES,
                    idrun=batch_idx,
                )
                # Normalize loss to account for gradient accumulation
                normalized_loss = loss / cfg.grad_accumulation_steps
                        
                # Backward pass
                if TRAIN_MODE:
                    normalized_loss.backward()

                # Store recent train metrics
                for metric_name, value in metrics.items():
                    if metric_name in recent_metrics:
                        recent_metrics[metric_name].append(value)

                # Compute gradient step index
                gradient_step_idx = log_count // cfg.grad_accumulation_steps
                log_count += 1

                # Push Metrics to W&B (every wandb_log_freq gradient steps)
                log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx

                smoothened_metrics = compute_smoothened_metrics(recent_metrics)
                if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                    log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)

                # [If applicable] Linearly warm up learning rate from 10% to 100% of original
                if cfg.lr_warmup_steps > 0:
                    lr_progress = min((gradient_step_idx + 1) / cfg.lr_warmup_steps, 1.0)  # Cap at 1.0
                    current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = current_lr

                if distributed_state.is_main_process and gradient_step_idx % cfg.wandb_log_freq == 0:
                    # Log the learning rate
                    # Make sure to do this AFTER any learning rate modifications (e.g., warmup/decay)
                    wandb.log(
                        {
                            "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                        },
                        step=log_step,
                    )

                # Optimizer and LR scheduler step
                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    progress.update()

                # Save model checkpoint: either keep latest checkpoint only or all checkpoints
                if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:            
                    save_training_checkpoint(
                        cfg=cfg,
                        run_dir=run_dir,
                        log_step=log_step,
                        vla=vla,
                        processor=processor,
                        pose_projector=pose_projector,
                        action_head=action_head,
                        distributed_state=distributed_state,
                    )

if __name__ == "__main__":
    train_omnivla()
