"""
finetune.py

Fine-tunes OpenVLA via LoRA.
"""
GPU_server = True
VISU = False

import sys
if GPU_server:
    sys.path.append("/home/noriaki/Learning-to-Drive-Anywhere-with-MBRA2/train")
    sys.path.append('/home/noriaki/lerobot')
    sys.path.append('/home/noriaki/map_cache')
    cast_loc = "/raid/users/noriaki/CAST_dataset/"

else:
    sys.path.append('/media/noriaki/Noriaki_Data/Learning-to-Drive-Anywhere-with-MBRA/train/')
    sys.path.append('/home/noriaki/Documents/map_cache')
    sys.path.append('/home/noriaki/Documents/lerobot')
    cast_loc = "/media/noriaki/Noriaki_Data/CAST_dataset/"

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
import numpy as np
import pickle

from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
    update_auto_map_MMN,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead, L1RegressionActionHead_idcat, L1RegressionDistHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForActionPrediction_Nav, PaddedCollatorForActionPrediction_Nav_MMN
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    IGNORE_INDEX,
)

from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset, LeLaN_Dataset_openvla, LeLaN_Dataset_openvla_act_MMN
from prismatic.vla.datasets import ViNTLeRobotDataset_IL2_gps_map2_crop_shadow_MMN, EpisodeSampler_IL_MMN, ViNT_Dataset_gps_MMN, BDD_Dataset_multi_MMN, CAST_Dataset_MMN
 
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import yaml
import json
import numpy
import random
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
from torchvision import transforms

print(sys.path)

from vint_train.models.exaug.exaug import ExAug_dist_delay
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
#from accelerate import init_empty_weights
from torch.utils.data import WeightedRandomSampler

os.environ["OMP_NUM_THREADS"] = "60"  # Set number of OpenMP threads
os.environ["MKL_NUM_THREADS"] = "60"  # Set number of MKL threads
torch.set_num_threads(60)  # Limit the number of CPU threads used by PyTorch

transform = ([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform = transforms.Compose(transform)

class WeightedDistributedSampler(DistributedSampler):
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.weights = weights
        self.replacement = replacement

    def __iter__(self):
        # Get indices for this rank
        indices = list(WeightedRandomSampler(
            self.weights, num_samples=self.num_samples, replacement=self.replacement
        ))
        # Only keep the indices for this GPU rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")      # Directory containing RLDS datasets
    dataset_name: str = "aloha_scoop_x_into_bowl"    # Name of fine-tuning dataset (e.g., `aloha_scoop_x_into_bowl`)
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    use_l1_regression: bool = True                   # If True, trains continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, trains continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 1e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10% to 100%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    #save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_freq: int = 1000                          # Checkpoint saving frequency in steps    
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
                                                     
    if GPU_server:                                                 
        #resume: bool = False                             # If True, resumes from checkpoint
        #resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
        resume: bool = True                             # If True, resumes from checkpoint
        #resume_step: Optional[int] = 35000                # (When `resume==True`) Step number that we are resuming from        
        resume_step: Optional[int] = 145000 
    else:
        #resume: bool = False                             # If True, resumes from checkpoint
        #resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from    
        resume: bool = True    
        resume_step: Optional[int] = 120000                # (When `resume==True`) Step number that we are resuming from
        #resume: bool = True    
        #resume_step: Optional[int] = 85000                # (When `resume==True`) Step number that we are resuming from
        #resume: bool = True    
        #resume_step: Optional[int] = 95000                # (When `resume==True`) Step number that we are resuming from
                        
    mmn_flag = True
         
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    diffusion_sample_freq: int = 50                  # (When `use_diffusion==True`) Frequency for sampling in steps

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

    # fmt: on


def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
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
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wrap a module with DistributedDataParallel.

    Args:
        module (nn.Module): PyTorch module.
        device_id (str): Device ID.
        find_unused (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of trainable parameters in a module.

    Args:
        module (nn.Module): PyTorch module.
        module_name (str): Name of model component.

    Returns:
        None.
    """
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a module, optionally loads checkpoint, moves to device, and wraps with DDP.

    Args:
        module_class (Type[nn.Module]): Class of PyTorch module to initialize.
        module_name (str): Name of model component to load checkpoint for.
        cfg (FinetuneConfig): Training configuration.
        device_id (str): Device ID.
        module_args (dict): Args for initializing the module.
        to_bf16 (bool): Whether to convert to torch.bfloat16 data type.
        find_unused_params (bool): Whether to detect parameters without gradients in distributed training.

    Returns:
        DistributedDataParallel: PyTorch module wrapped with DDP.
    """
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume and module_name != "dist_head":
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
    #state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
    #module.load_state_dict(state_dict)
        
    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)

def sinc_apx(angle):
    return torch.sin(3.141592*angle + 0.000000001)/(3.141592*angle + 0.000000001)

def twist_to_pose_diff_torch(v, w, dt):
    """integrate 2D twist to get pose difference.

    Assuming constant velocity during time period `dt`.

    Args:
        v (float): velocity
        w (float): angular velocity
        dt (float): time delta

    """

    theta = -w  * dt
    z = v * dt * sinc_apx(-theta / numpy.pi)
    x = -v * dt * sinc_apx(-theta / (2 * numpy.pi)) * torch.sin(-theta / 2)
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
    dist_head,
    mbra,
    noisy_action_projector,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_diffusion,
    use_proprio,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
    mode="vali",
    idrun=0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute model forward pass and metrics for both training and validation.

    Args:
        vla (OpenVLAForActionPrediction_MMN): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        use_l1_regression (bool): Whether to use L1 regression.
        use_diffusion (bool): Whether to use diffusion.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.
        num_patches (int): Number of vision patches.
        compute_diffusion_l1 (bool): Whether to sample actions and compute L1 loss for diffusion (do this once every
                                    diffusion_sample_freq steps during training; do it every batch for validation)
        num_diffusion_steps_train (int): Number of diffusion steps for training (only used for diffusion).

    Returns:
        tuple: (loss, metrics_dict)
            loss: The loss tensor with gradient for backpropagation.
            metrics_dict: Dictionary of computed metrics (detached values for logging).
    """
    metrics = {}
    context_size = 5

    # Get ground-truth action labels
    #print("batch[actions]", batch["actions"])
    ground_truth_actions = batch["actions"].to(device_id).to(torch.bfloat16)
    nomad_actions = batch["actions_nomad"].to(device_id).to(torch.bfloat16)
    
    # [Only for diffusion] Sample noisy actions used as input for noise predictor network
    if use_diffusion:
        noisy_dict = action_head.module.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, diffusion_timestep_embeddings = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["diffusion_timestep_embeddings"],
        )
    else:
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

    Blan = batch["cur_image"].size()[0]
    # VLA forward pass
    #print(batch["cur_image_crop"].size())
    #print(batch["cur_image_crop"][:, 3*context_size:].size())
    goal_img_cat = torch.cat((transform(batch["cur_image_crop"][:, 3*context_size:]), transform(batch["goal_image_crop"])), axis=1)
    map_img_cat = torch.cat((transform(batch["cur_map_image"]), transform(batch["goal_map_image"]), transform(batch["cur_image_crop"][:, 3*context_size:])), axis=1) #dummy
    img_hist = torch.split(batch["cur_image"], 3, dim=1)
    img_hist_norm_list = [transform(obs_image) for obs_image in img_hist]
    img_hist_norm = torch.concat(img_hist_norm_list, dim=1)      
    
    img_hist_crop = torch.split(batch["cur_image_crop"], 3, dim=1)
    img_hist_crop_norm_list = [transform(obs_image) for obs_image in img_hist_crop]
    img_hist_crop_norm = torch.concat(img_hist_crop_norm_list, dim=1)    

    rsize = 0.3*torch.ones(Blan, 1, 1).to(device_id)
    delay = torch.zeros(Blan, 1, 1).to(device_id)
    linear_vel_old = 0.5*torch.ones(Blan, 6).float().to(device_id)
    angular_vel_old = 0.0*torch.ones(Blan, 6).float().to(device_id)
    vel_past = torch.cat((linear_vel_old, angular_vel_old), axis=1).unsqueeze(2)          
                
    #print(img_hist_norm.size())
    #mbra.eval()
    with torch.no_grad():
        linear_vel, angular_vel, _ = mbra(img_hist_norm, transform(batch["goal_image_8"]), rsize, delay, vel_past)
    #print("linear_vel", linear_vel)
    #print("angular_vel", angular_vel)    
    action_mbra = robot_pos_model(linear_vel, angular_vel)  
    
    #print("batch[goal_pose]", batch["goal_pose"].size())
    modality_id = batch["goal_mask_select"]
    #print("modality_id", modality_id)
    if GPU_server:
        with torch.autocast("cuda", dtype=torch.bfloat16):    
        #with torch.autocast("cuda", dtype=torch.float8):
            output: CausalLMOutputWithPast = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                attention_mask_label=batch["attention_mask_label"].to(device_id),                  
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                #pixel_values_goal=goal_img_cat.to(torch.bfloat16).to(device_id),
                #img_hist=img_hist_crop_norm.to(torch.bfloat16).to(device_id),
                #map_images=map_img_cat.to(torch.bfloat16).to(device_id),
                #goal_pose=batch["goal_pose"].to(torch.bfloat16).to(device_id),
                modality_id=modality_id.to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                #proprio=batch["proprio"] if use_proprio else None,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id) if use_proprio else None,                
                proprio_projector=proprio_projector if use_proprio else None,
                noisy_actions=noisy_actions if use_diffusion else None,
                noisy_action_projector=noisy_action_projector if use_diffusion else None,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
                use_film=use_film,
            )
    else:
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    attention_mask_label=batch["attention_mask_label"].to(device_id),                    
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    #pixel_values_goal=goal_img_cat.to(torch.bfloat16).to(device_id),
                    #img_hist=img_hist_norm.to(torch.bfloat16).to(device_id),
                    #map_images=map_img_cat.to(torch.bfloat16).to(device_id),
                    #goal_pose=batch["goal_pose"].to(torch.bfloat16).to(device_id),                     
                    modality_id=modality_id.to(torch.bfloat16).to(device_id),                                       
                    labels=batch["labels"],
                    output_hidden_states=True,
                    #proprio=batch["proprio"] if use_proprio else None,                    
                    proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id) if use_proprio else None,
                    proprio_projector=proprio_projector if use_proprio else None,
                    noisy_actions=noisy_actions if use_diffusion else None,
                    noisy_action_projector=noisy_action_projector if use_diffusion else None,
                    diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
                    use_film=use_film,
                )
    
    # Get object pose
    #print(batch.keys())
    goal_pose_lelan = batch["goal_pose"].to(device_id)
    obj_pose_norm_lelan = batch["obj_pose_norm"].to(dtype=torch.bfloat16).to(device_id)
    #print("run_forward_pass", pose_obj_lelan.size(), pose_obj_lelan)        
    # Get action masks needed for logging
    #print("batch[labels]", batch["labels"].size(), batch["input_ids"].size())
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    #print("batch[labels][:, 1:]", batch["labels"][:, 1:])
    #print("ground_truth_token_ids", ground_truth_token_ids)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

    # Compute metrics for discrete action representation (next-token prediction)
    if not (use_l1_regression or use_diffusion):
        loss = output.loss
        predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)
        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
        )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
        )
        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "curr_action_accuracy": curr_action_accuracy.item(),
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_accuracy": next_actions_accuracy.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
            }
        )
    # Compute metrics for continuous action representations (L1 regression | diffusion)
    else:
        #print("kocchi dayone??")
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        #print(last_hidden_states.size())
        # Get hidden states for text portion of prompt+response (after the vision patches)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        # Get hidden states for action portion of response
        batch_size = batch["input_ids"].shape[0]
        #print(last_hidden_states.size(), text_hidden_states.size(), current_action_mask.size(), current_action_mask.sum(), next_actions_mask.size(), next_actions_mask.sum(), ground_truth_token_ids.size())
        #print(text_hidden_states.size(), current_action_mask.size(), current_action_mask.sum(), next_actions_mask.size(), next_actions_mask.sum())
        #print(text_hidden_states[current_action_mask | next_actions_mask].size(), text_hidden_states.size(), current_action_mask.size(), current_action_mask.sum(), next_actions_mask.size(), next_actions_mask.sum())
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)
        #print(last_hidden_states.size(), text_hidden_states.size(), actions_hidden_states.size())  

        if use_l1_regression:
            # Predict action
            #print("actions_hidden_states", actions_hidden_states.size(), modality_id.size(), modality_id.type())
            if GPU_server:
                predicted_actions = action_head.module.predict_action(actions_hidden_states, modality_id.to(torch.bfloat16).to(device_id))
                predicted_dist = dist_head.module.predict_action(actions_hidden_states)
            else:
                with torch.no_grad():
                #if True:
                    predicted_actions = action_head.module.predict_action(actions_hidden_states, modality_id.to(torch.bfloat16).to(device_id))
                    predicted_dist = dist_head.module.predict_action(actions_hidden_states)
                                
            #print("predicted_dist", predicted_dist.size())
            #print("temp_dist", batch["temp_dist"].size(), batch["temp_dist"])
            # Get full L1 loss
            #print(predicted_actions.size(), ground_truth_actions.size(), goal_pose_lelan.size(), obj_pose_norm_lelan.size())
            #loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)
            
            #print("ground_truth_actions dtype:", ground_truth_actions.dtype)
            #print("predicted_actions dtype:", predicted_actions.dtype)
            #print("obj_pose_norm_lelan dtype:", obj_pose_norm_lelan.dtype)
            
            #lan_bool = batch["goal_mask_select"] == 7
            lan_bool = (batch["goal_mask_select"] == 7)|(batch["goal_mask_select"] == 8)
            mask_lan = lan_bool.to(torch.bfloat16).to(device_id).unsqueeze(1).unsqueeze(2).repeat(1,8,4)
            mask_notlan = -1.0*(mask_lan - 1.0)
            
            mask_act = batch["action_mask_select"].to(torch.bfloat16).to(device_id).unsqueeze(1).unsqueeze(2).repeat(1,8,4)
            mask_notact = -1.0*(mask_act - 1.0)
            
            #print("goal_mask_select", batch["goal_mask_select"])            
            #print("mask_lan", mask_lan)
            #print("mask_notlan", mask_notlan)
            #print("mask_act", mask_act)
            #print("mask_notact", mask_notact)                                    
            action_raw_mbra = mask_act*ground_truth_actions + mask_notact*action_mbra.detach().to(torch.bfloat16)
            action_label_lan = mask_notlan*action_raw_mbra + mask_lan*nomad_actions
            #print("action_label_lan", action_label_lan)
            
            #mask_0 = (batch["temp_dist"] == 0).to(torch.bfloat16).to(device_id).unsqueeze(1).unsqueeze(2).repeat(1,8,4)
            #mask_non0 = (batch["temp_dist"] != 0).to(torch.bfloat16).to(device_id).unsqueeze(1).unsqueeze(2).repeat(1,8,4)
            #action_label_lan = mask_non0*action_mbra.detach().to(torch.bfloat16)+ mask_0*ground_truth_actions.detach()
            #distance imitation
            limited_temp_dist = torch.clip(batch["temp_dist"], min=0.0, max=20.0) 
            #loss = 1.0*torch.nn.L1Loss()(ground_truth_actions, predicted_actions) + 0.1*torch.nn.L1Loss()(obj_pose_norm_lelan, predicted_actions[:,-1,0:2]) + 0.1*torch.nn.L1Loss()(predicted_actions[:,0:-1], predicted_actions[:,1:])
            #loss = 1.0*torch.nn.MSELoss()(predicted_dist, limited_temp_dist.to(dtype=torch.bfloat16).to(device_id)) + 1.0*torch.nn.MSELoss()(action_label_lan, predicted_actions) + 0.1*torch.nn.MSELoss()(obj_pose_norm_lelan[lan_bool], predicted_actions[:,-1,0:2][lan_bool]) + 0.1*torch.nn.MSELoss()(predicted_actions[:,0:-1], predicted_actions[:,1:])
            loss = 0.0*torch.nn.MSELoss()(predicted_dist, limited_temp_dist.to(dtype=torch.bfloat16).to(device_id)) + 1.0*torch.nn.MSELoss()(action_label_lan, predicted_actions) + 0.1*torch.nn.MSELoss()(obj_pose_norm_lelan[lan_bool], predicted_actions[:,-1,0:2][lan_bool]) + 0.1*torch.nn.MSELoss()(predicted_actions[:,0:-1], predicted_actions[:,1:])            
            L1_action = torch.nn.L1Loss()(action_label_lan, predicted_actions)
            L1_obj = torch.nn.L1Loss()(obj_pose_norm_lelan, predicted_actions[:,-1,0:2])
            L1_smooth = torch.nn.L1Loss()(predicted_actions[:,0:-1], predicted_actions[:,1:])
            L2_action = torch.nn.MSELoss()(action_label_lan, predicted_actions)
            #print("L2_action", L2_action)
            L2_obj = torch.nn.MSELoss()(obj_pose_norm_lelan[lan_bool], predicted_actions[:,-1,0:2][lan_bool])
            L2_smooth = torch.nn.MSELoss()(predicted_actions[:,0:-1], predicted_actions[:,1:])
            L2_dist = torch.nn.MSELoss()(predicted_dist, limited_temp_dist.to(dtype=torch.bfloat16).to(device_id))
            #print("L2_obj", L2_obj, "L2_action", L2_action, "L2_smooth", L2_smooth)
            
            loss_list = []
            task_list = []
            for icl in range(9):
                mask_task = batch["goal_mask_select"] == icl
                #print(mask_task)
                L2_action_task = torch.nn.MSELoss()(action_label_lan[mask_task], predicted_actions[mask_task])
                loss_list.append(L2_action_task)
                task_list.append(torch.sum(mask_task.float()))
            #print(task_list)    
            #print(torch.sum(lan_bool))
                        
        if use_diffusion:
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)
            # Get diffusion noise prediction MSE loss
            noise_pred = noise_pred.reshape(noise.shape)
            loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")

            # Only sample actions and compute L1 losses if specified
            if compute_diffusion_l1:
                with torch.no_grad():
                    predicted_actions = run_diffusion_sampling(
                        vla=vla,
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio_projector=proprio_projector,
                        batch=batch,
                        batch_size=batch_size,
                        num_patches=num_patches,
                        actions_shape=ground_truth_actions.shape,
                        device_id=device_id,
                        current_action_mask=current_action_mask,
                        next_actions_mask=next_actions_mask,
                        use_proprio=use_proprio,
                        use_film=use_film,
                    )

        metrics.update(
            {
                "loss_value": loss.item(),  # Detached value for logging
                "L1_action_value": L1_action.item(),  # Detached value for logging                
                "L1_obj_value": L1_obj.item(),  # Detached value for logging
                "L1_smooth_value": L1_smooth.item(),  # Detached value for logging
                "L2_action_value": L2_action.item(),  # Detached value for logging                
                "L2_obj_value": L2_obj.item(),  # Detached value for logging
                "L2_smooth_value": L2_smooth.item(),  # Detached value for logging       
                "L2_dist_value": L2_dist.item(),  # Detached value for logging                  
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

        # Get detailed L1 losses for logging
        should_log_l1_loss = not use_diffusion or (use_diffusion and compute_diffusion_l1)
        #print("should_log_l1_loss", should_log_l1_loss)
        if should_log_l1_loss:
            ground_truth_curr_action = ground_truth_actions[:, 0]
            predicted_curr_action = predicted_actions[:, 0]
            ground_truth_next_actions = ground_truth_actions[:, 1:]
            predicted_next_actions = predicted_actions[:, 1:]
            curr_action_l1_loss = torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action)
            next_actions_l1_loss = torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions)
            metrics.update(
                {
                    "curr_action_l1_loss": curr_action_l1_loss.item(),
                    "next_actions_l1_loss": next_actions_l1_loss.item(),
                }
            )
        #print("ground_truth_actions", ground_truth_actions.detach().cpu())
        #batch_viz_obs_images_lan = TF.resize((255.0*obs_images_lan).type(torch.uint8), VISUALIZATION_IMAGE_SIZE[::-1])
        #print(mode, batch["img_PIL"])        
        if idrun == 0 and mode == "vali":
            if GPU_server:
                visualize_lelan_train(
                    batch["img_PIL"],
                    batch["cur_image_crop"][:, 3*context_size:],
                    batch["cur_image"][:, 3*context_size:],
                    batch["goal_image_crop"],
                    batch["goal_image_8"],
                    #batch_viz_goal_images_lan,                   
                    obj_pose_norm_lelan.detach().cpu(),
                    ground_truth_actions.detach().cpu(),
                    action_mbra.detach().cpu(),
                    nomad_actions.detach().cpu(),  
                    action_label_lan.detach().cpu(),                      
                    predicted_actions.detach().cpu(), 
                    batch["goal_mask_select"],                
                    "train",   
                    0,                    
                    10,                                                    
                    )
            else:
                visualize_lelan_train(
                    batch["img_PIL"],
                    batch["cur_image_crop"][:, 3*context_size:],
                    batch["cur_image"][:, 3*context_size:],
                    batch["goal_image_crop"],
                    batch["goal_image_8"],              
                    batch["goal_pose"].detach().cpu(),#obj_pose_norm_lelan.detach().cpu(),
                    ground_truth_actions.detach().cpu(),  
                    action_mbra.detach().cpu(),
                    nomad_actions.detach().cpu(),  
                    action_label_lan.detach().cpu(),                    
                    predicted_actions.detach().cpu(),                 
                    batch["goal_mask_select"],
                    "train",   
                    0,                    
                    10,                                                    
                    )  

        elif VISU == True:
            visualize_lelan_eval(
                batch["img_PIL"],
                batch["cur_image_crop"][:, 3*context_size:],
                batch["cur_image"][:, 3*context_size:],
                batch["goal_image_crop"],
                batch["goal_image_8"],               
                obj_pose_norm_lelan.detach().cpu(),   
                batch["goal_pose"].detach().cpu(),
                ground_truth_actions.detach().cpu(),
                action_mbra.detach().cpu(),    
                predicted_actions.detach().cpu(),   
                action_label_lan.detach().cpu(),    
                batch["goal_mask_select"],          
                "train",   
                0,         
                idrun,                              
                1,                  
                False,#True: lelan, False GNM                               
                )                                        

    # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
    return loss, metrics


def run_diffusion_sampling(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    batch_size,
    num_patches,
    actions_shape,
    device_id,
    current_action_mask,
    next_actions_mask,
    use_proprio,
    use_film,
) -> torch.Tensor:
    """
    Run diffusion sampling (reverse diffusion) to generate actions.

    Args:
        vla (OpenVLAForActionPrediction_MMN): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        batch (dict): Input batch.
        batch_size (int): Batch size.
        num_patches (int): Number of vision patches.
        actions_shape (tuple): Shape of ground-truth actions.
        device_id (str): Device ID.
        current_action_mask (torch.Tensor): Mask for current action.
        next_actions_mask (torch.Tensor): Mask for next actions.
        use_proprio (bool): Whether to use proprioceptive state as input.
        use_film (bool): Whether to use FiLM for better language following.

    Returns:
        torch.Tensor: Predicted actions.
    """
    # Sample random noisy action, used as the starting point for reverse diffusion
    noise = torch.randn(
        size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
        device=device_id,
        dtype=torch.bfloat16,
    )  # (B, chunk_len, action_dim)

    # Set diffusion timestep values
    action_head.module.noise_scheduler.set_timesteps(action_head.module.num_diffusion_steps_train)

    # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
    curr_noisy_actions = noise
    for t in action_head.module.noise_scheduler.timesteps:
        # Get diffusion model's noise prediction (conditioned on VLA latent embedding, current noisy action embedding,
        # and diffusion timestep embedding)
        timesteps = torch.Tensor([t]).repeat(batch_size).to(device_id)
        diffusion_timestep_embeddings = (
            action_head.module.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
        )  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                labels=batch["labels"],
                output_hidden_states=True,
                proprio=batch["proprio"] if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                noisy_actions=curr_noisy_actions,
                noisy_action_projector=noisy_action_projector,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                use_film=use_film,
            )
            # Get last layer hidden states
            last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
            # Get hidden states for text portion of prompt+response (after the vision patches)
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            # Get hidden states for action portion of response
            actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
                batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1
            )  # (B, act_chunk_len, D)
            actions_hidden_states = actions_hidden_states.to(torch.bfloat16)
            # Predict noise
            noise_pred = action_head.module.predict_noise(actions_hidden_states)

        # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
        curr_noisy_actions = action_head.module.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

    return curr_noisy_actions.reshape(actions_shape)


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
    proprio_projector,
    noisy_action_projector,
    action_head,
    dist_head,
    train_dataset,
    distributed_state,
) -> None:
    """
    Save all training checkpoints including model components, LoRA adapter, and dataset statistics.

    Args:
        cfg (FinetuneConfig): Training configuration.
        run_dir (Path): Experiment run directory path.
        log_step (int): Current logging step.
        vla (OpenVLAForActionPrediction_MMN): Vision-language-action policy.
        processor (PrismaticProcessor): OpenVLA inputs processor.
        proprio_projector (nn.Module): Proprioceptive state projector module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        action_head (nn.Module): Action head module.
        train_dataset (RLDSDataset): Training dataset.
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
        #save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    # Wait for directories to be created
    dist.barrier()

    # Save model components (main process only)
    if distributed_state.is_main_process:
        # Save processor and LoRA adapter
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)

        # Save other components
        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(), checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}"
            )

        if (cfg.use_l1_regression or cfg.use_diffusion) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")
            torch.save(dist_head.state_dict(), checkpoint_dir / f"dist_head--{checkpoint_name_suffix}")
            
        if cfg.use_film:
            # To be safe, just save the entire vision backbone (not just FiLM components)
            torch.save(
                vla.module.vision_backbone.state_dict(), checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}"
            )

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

def to_numpy(tensor: torch.Tensor) -> numpy.ndarray:
    return tensor.detach().cpu().to(torch.float32).numpy()

def visualize_lelan_train(
    batch_viz_obs_images_lan: torch.Tensor,
    batch_current_crop: torch.Tensor,
    batch_current: torch.Tensor,
    batch_goal_crop: torch.Tensor,
    batch_goal: torch.Tensor,  
    goal_pos_lan: torch.Tensor,
    traj_raw: torch.Tensor,
    traj_mbra: torch.Tensor,
    traj_nomad: torch.Tensor,    
    traj_select: torch.Tensor,
    est_traj: torch.Tensor,
    goal_mask_select: torch.Tensor,
    #project_folder: str,
    eval_type: str,    
    epoch: int,
    num_images_log: int = 10,    
    #use_wandb: bool = True,      
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

    #num_images_log = min(num_images_log, batch_viz_obs_images_lan[0].shape[0])      
    num_images_log = min(num_images_log, batch_current_crop.size()[0])
        
    wandb_list = []

    for i in range(num_images_log):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])      
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])   
        ax_obc = fig.add_subplot(gs[1:2, 1:2])
        ax_goalc = fig.add_subplot(gs[1:2, 2:3])

        #obs_image = to_numpy(batch_viz_obs_images_lan[i][-3:])
        #obs_image = np.moveaxis(obs_image, 0, -1)                    
        #goal_image = to_numpy(batch_viz_goal_images_lan[i])
        #goal_image = np.moveaxis(goal_image, 0, -1)    
        obs_image_crop = to_numpy(255.0*batch_current_crop[i])
        obs_image_crop = numpy.moveaxis(obs_image_crop, 0, -1)                    
        goal_image_crop = to_numpy(255.0*batch_goal_crop[i])
        goal_image_crop = numpy.moveaxis(goal_image_crop, 0, -1)    
        obs_image = to_numpy(255.0*batch_current[i])
        obs_image = numpy.moveaxis(obs_image, 0, -1)                    
        goal_image = to_numpy(255.0*batch_goal[i])
        goal_image = numpy.moveaxis(goal_image, 0, -1)    
                
        #batch_viz_obs_images_lan = batch_viz_obs_images_lan[i]
        #ax_ob.imshow(batch_viz_obs_images_lan)                      
        #ax_ob.imshow((batch_viz_obs_images_lan).astype(np.uint8))               
        #ax_goal.imshow((goal_image).astype(np.uint8))     
        ax_ob.imshow((obs_image).astype(numpy.uint8))   
        ax_goal.imshow((goal_image).astype(numpy.uint8))              
        ax_obc.imshow((obs_image_crop).astype(numpy.uint8))   
        ax_goalc.imshow((goal_image_crop).astype(numpy.uint8))
                                                    
        xgt = to_numpy(goal_pos_lan[i,0])
        ygt = to_numpy(goal_pos_lan[i,1])
        task_id = goal_mask_select[i].item()
                                
        x_nomad = traj_nomad[i, :, 0].detach().cpu().to(torch.float32).numpy()
        y_nomad = traj_nomad[i, :, 1].detach().cpu().to(torch.float32).numpy()
        x_mbra = traj_mbra[i, :, 0].detach().cpu().to(torch.float32).numpy()
        y_mbra = traj_mbra[i, :, 1].detach().cpu().to(torch.float32).numpy()    
        x_raw = traj_raw[i, :, 0].detach().cpu().to(torch.float32).numpy()
        y_raw = traj_raw[i, :, 1].detach().cpu().to(torch.float32).numpy()
        x_select = traj_select[i, :, 0].detach().cpu().to(torch.float32).numpy()
        y_select = traj_select[i, :, 1].detach().cpu().to(torch.float32).numpy()          
            
        x_est = est_traj[i, :, 0].detach().cpu().to(torch.float32).numpy()
        y_est = est_traj[i, :, 1].detach().cpu().to(torch.float32).numpy()

        ax_graph.plot(-y_select, x_select, marker = 'o', color='m', linewidth=4, markersize=10, label="select") 
        ax_graph.plot(-y_raw, x_raw, marker = 'o', color='c', label="raw(GNM) or mbra(BDD)")                                                          
        ax_graph.plot(-y_nomad, x_nomad, marker = 'o', color='red', label="nomad")        
        ax_graph.plot(-y_mbra, x_mbra, marker = 'o', color='green', label="mbra")          
        ax_graph.plot(-y_est, x_est, marker = 'o', color='blue', label="est.")                                        
        ax_graph.plot(-ygt, xgt, marker = '*', color='red')   
        ax_graph.text(2.5, -0.2, str(task_id))
                      
        # set title
        ax_graph.set_title(f"est. trajectory (normzlied dim.)")
        ax_graph.set_xlim(-10.0, 10.0)
        ax_graph.set_ylim(-1.0, 16.0)
        ax_graph.legend(loc='best')                  
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"goal")
        ax_obc.set_title(f"cropped observation")
        ax_goalc.set_title(f"cropped goal")    
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_lelan_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))        
        plt.close(fig)

def visualize_lelan_eval(
    batch_viz_obs_images_lan: torch.Tensor,
    batch_current_crop: torch.Tensor,
    batch_current: torch.Tensor,
    batch_goal_crop: torch.Tensor,
    batch_goal: torch.Tensor,
    #batch_viz_goal_images_lan: torch.Tensor,     
    goal_pos_lan: torch.Tensor, 
    goal_pos: torch.Tensor, 
    traj_nomad: torch.Tensor,
    traj_mbra: torch.Tensor,    
    est_traj: torch.Tensor,
    select_traj: torch.Tensor,    
    #project_folder: str,
    goal_mask_select: torch.Tensor,
    eval_type: str,    
    epoch: int,
    count: int,
    num_images_log: int = 10,            
    #use_wandb: bool = True,  
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

    #num_images_log = min(num_images_log, batch_viz_obs_images_lan[0].shape[0])      
    #num_images_log = min(num_images_log, batch_viz_obs_images_lan[0].shape[0])
        
    wandb_list = []
    #print(type(batch_viz_obs_images_lan))
    PIL_list = batch_viz_obs_images_lan
    
    if lan:
        lan_id = 0
        goal_pos_gt = goal_pos_lan
    else:
        lan_id = 1
        goal_pos_gt = goal_pos
    
    for i in range(num_images_log):
        i = lan_id
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2,3)
        ax_graph = fig.add_subplot(gs[0:2, 0:1])      
        ax_ob = fig.add_subplot(gs[0:1, 1:2])
        ax_goal = fig.add_subplot(gs[0:1, 2:3])   
        ax_obc = fig.add_subplot(gs[1:2, 1:2])
        ax_goalc = fig.add_subplot(gs[1:2, 2:3])

        obs_image_crop = to_numpy(255.0*batch_current_crop[i])
        obs_image_crop = numpy.moveaxis(obs_image_crop, 0, -1)                    
        goal_image_crop = to_numpy(255.0*batch_goal_crop[i])
        goal_image_crop = numpy.moveaxis(goal_image_crop, 0, -1)    
        obs_image = to_numpy(255.0*batch_current[i])
        obs_image = numpy.moveaxis(obs_image, 0, -1)                    
        goal_image = to_numpy(255.0*batch_goal[i])
        goal_image = numpy.moveaxis(goal_image, 0, -1)    
                
        #batch_viz_obs_images_lan = batch_viz_obs_images_lan
        
        #batch_viz_obs_images_lan = Image.fromarray(batch_viz_obs_images_lan[i].astype(np.uint8))
        #ax_ob.imshow(PIL_list[i])                
        ax_ob.imshow((obs_image).astype(numpy.uint8))   
        ax_goal.imshow((goal_image).astype(numpy.uint8))              
        ax_obc.imshow((obs_image_crop).astype(numpy.uint8))   
        ax_goalc.imshow((goal_image_crop).astype(numpy.uint8))         
                                            
        xgt = to_numpy(goal_pos_gt[i,0])
        ygt = to_numpy(goal_pos_gt[i,1])
        task_id = goal_mask_select[i].item()
            
        x_nomad = traj_nomad[i, :, 0].detach().cpu().to(torch.float32).numpy()
        y_nomad = traj_nomad[i, :, 1].detach().cpu().to(torch.float32).numpy()
        x_mbra = traj_mbra[i, :, 0].detach().cpu().to(torch.float32).numpy()
        y_mbra = traj_mbra[i, :, 1].detach().cpu().to(torch.float32).numpy()          
        x_est = est_traj[i, :, 0].detach().cpu().to(torch.float32).numpy()
        y_est = est_traj[i, :, 1].detach().cpu().to(torch.float32).numpy()
        x_select = select_traj[i, :, 0].detach().cpu().to(torch.float32).numpy()
        y_select = select_traj[i, :, 1].detach().cpu().to(torch.float32).numpy()

        ax_graph.plot(-y_select, x_select, marker = 'o', color='m', linewidth=4, markersize=10, label="select") 
        ax_graph.plot(-y_est, x_est, marker = 'o', color='blue', label="est.")                                                          
        ax_graph.plot(-y_nomad, x_nomad, marker = 'o', color='red', label="nomad")
        ax_graph.plot(-y_mbra, x_mbra, marker = 'o', color='green', label="mbra")                                                
        ax_graph.plot(-ygt, xgt, marker = '*', color='red')   
        ax_graph.text(2.5, -0.2, str(task_id))
                                                   
        # set title
        ax_graph.set_title(f"est. trajectory (normzlied dim.)")
        ax_graph.set_xlim(-10.0, 10.0)
        ax_graph.set_ylim(-1.0, 16.0)
        ax_graph.legend(loc='best')                  
        ax_ob.set_title(f"observation")
        ax_goal.set_title(f"goal")
        ax_obc.set_title(f"cropped observation")
        ax_goalc.set_title(f"cropped goal")                
        #ax_past.set_title(f"velocity command")
                        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)
        
        save_path = os.path.join(visualize_path, f"sample_lelan_{count}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))        
        plt.close(fig)

def run_validation(
    vla,
    action_head,
    dist_head,
    mbra,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    """
    Compute validation set metrics for logging.

    Args:
        vla (OpenVLAForActionPrediction_MMN): Vision-language-action policy.
        action_head (nn.Module): Action head module.
        noisy_action_projector (nn.Module): Noisy action projector module (only used for diffusion).
        proprio_projector (nn.Module): Proprioceptive state projector module.
        val_dataloader (DataLoader): Validation data loader.
        action_tokenizer (ActionTokenizer): Action tokenizer.
        device_id (str): Device ID.
        cfg (FinetuneConfig): Training configuration.
        num_patches (int): Number of vision patches.
        log_step (int): Current logging step.
        distributed_state (PartialState): Distributed training state.
        val_time_limit (int): Time limit for computing validation metrics.

    Returns:
        None.
    """
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0

    # List to store validation metrics
    all_val_metrics = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            # Always compute L1 loss for validation, even for diffusion
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                dist_head=dist_head,
                mbra=mbra,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_diffusion=cfg.use_diffusion,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                mode="vali",
                idrun=val_batches_count,
            )

            # Add the loss value to the metrics
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1

            # Cut testing on validation set short if it exceeds time limit
            if time.time() - val_start_time > val_time_limit:
                break

    # Compute average validation metrics
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [metrics[metric_name] for metrics in all_val_metrics if metric_name in metrics]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    # Add batch count to metrics
    avg_val_metrics["val_batches_count"] = val_batches_count

    # Log validation metrics to W&B
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)

"""
def merge_batches(batch_list):
    merged = {}
    keys = batch_list[0].keys()
    
    for key in keys:
        print(key)
        if key not in ["proprio"]:
            merged[key] = torch.cat([batch[key] for batch in batch_list], dim=0)
    
    return merged
"""
def merge_batches_padding(batch_list, pad_token_id, IGNORE_INDEX, model_max_length):
    """
    Merge a list of dictionary batches into a single dictionary,
    concatenating tensor values along the batch dimension (dim=0).
    Non-tensor values are skipped.
    batch_list is the list of batch data. Order is [lelan, gnm, frod, bdd]
    """
    merged = {}
    keys = batch_list[0].keys()

    #merged_torch_keys = []  
    #merged_list_keys = []        
    #unmerged_keys = []       
    for key in keys:
        #print(key)
        #if key == "input_ids":
        #    print(batch_list[0][key])
        #    print(batch_list[1][key])
                        
        values = [batch[key] for batch in batch_list]
        first_value = values[0]
        if key == "temp_dist":
            batch_size  = [batch[key].size()[0] for batch in batch_list]
            batch_tdist = [batch[key] for batch in batch_list]

        if isinstance(first_value, torch.Tensor):
            #merged_torch_keys.append(key)
            merged[key] = torch.cat(values, dim=0)
        elif isinstance(first_value, list):
            #merged_list_keys.append(key)
            combined_list = []
            for v in values:
                combined_list.extend(v)
            merged[key] = combined_list            
        else:
            #unmerged_keys.append(key)
            # Optionally log or keep one example
            # print(f"Skipping non-tensor key: {key}")
            pass  # or merged[key] = batch_list[0][key]

    input_ids = pad_sequence(merged["input_ids"], batch_first=True, padding_value=pad_token_id)
    merged["input_ids"] = input_ids[:, : model_max_length]
    labels = pad_sequence(merged["labels"], batch_first=True, padding_value=IGNORE_INDEX)
    merged["labels"] = labels[:, : model_max_length]
    merged["attention_mask"] = merged["input_ids"].ne(pad_token_id)        
    merged["attention_mask_label"] = merged["labels"].ne(IGNORE_INDEX)            
    #print(merged["input_ids"].size(), merged["labels"].size(), merged["attention_mask"].size())


    #print("batch size", batch_size)
    #print("temp distance", batch_tdist)
    goal_mask_bdd = []
    goal_mask_frod = []
    goal_mask_gnm = []
    goal_mask_lan = [] 
    goal_mask_cast = []        
    Blan = batch_size[0]
    Bsub = batch_size[1]
    B = batch_size[2]
    Bbdd = batch_size[3] 
    Bcast = batch_size[4]      
    distance_lan = batch_tdist[0]
    dist_label_sub = batch_tdist[1]
    distance = batch_tdist[2] 
    distance_bdd = batch_tdist[3]           
    distance_cast = batch_tdist[4]              
    """
    my_list = [7,8]            
    for idl in range(Blan):
        if batch["temp_dist"][idl] == 0:
            if random.random() > 0.5:
                goal_mask_lan.append(7)
            else:
                goal_mask_lan.append(random.choice(my_list))   
        else:
            goal_mask_lan.append(6)   
    """
    
    #my_list = [4,7,8]
    my_list = [7,8]    
    for idl in range(Blan):
        if distance_lan[idl] == 0:
            if random.random() > 0.5:
                goal_mask_lan.append(random.choice(my_list))   
            else:
                #goal_mask_lan.append(8)   
                goal_mask_lan.append(7)   
        else:
            #goal_mask_lan.append(random.randint(6,7))  
            goal_mask_lan.append(6)
                                
    #for idf in range(B):
    #    if distance[idf] <= 20:
    #        goal_mask_frod.append(random.randint(0,6))
    #    else:
    #        goal_mask_frod.append(random.randint(0,5))
    for idf in range(B):
        if distance[idf] <= 20:
            goal_mask_frod.append(random.randint(4,6))
        else:
            goal_mask_frod.append(random.randint(4,5))                    
                                        
    for idg in range(Bsub):
        if dist_label_sub[idg] <= 20:
            goal_mask_gnm.append(random.randint(4,6))
        else:
            goal_mask_gnm.append(random.randint(4,5))
                    
    for idt in range(Bbdd):
        if distance_bdd[idt] <= 20:
            goal_mask_bdd.append(random.randint(4,6))
        else:
            goal_mask_bdd.append(random.randint(4,5))

    for idt in range(Bcast):
        goal_mask_cast.append(7)
                  
    """
    #my_list = [4,7,8]
    #my_list = [7,8]    
    for idl in range(Blan):
        goal_mask_lan.append(6)
                                
    for idf in range(B):
        if distance[idf] <= 20:
            goal_mask_frod.append(random.randint(4,6))
        else:
            goal_mask_frod.append(random.randint(4,5))
                    
    for idg in range(Bsub):
        if dist_label_sub[idg] <= 20:
            goal_mask_gnm.append(random.randint(4,6))
        else:
            goal_mask_gnm.append(random.randint(4,5))
                    
    for idt in range(Bbdd):
        if distance_bdd[idt] <= 20:
            goal_mask_bdd.append(random.randint(4,6))
        else:
            goal_mask_bdd.append(random.randint(4,5))    
    """
    #goal_mask_all = []   
    #for isum in range(Blan + B + Bsub + Bbdd):
    #    goal_mask_all.append(4)
    #    
    # [lelan, gnm, frod, bdd]  
    #goal_mask_select = torch.tensor(goal_mask_all)   
    goal_mask_select = torch.tensor(goal_mask_lan + goal_mask_gnm + goal_mask_frod + goal_mask_bdd + goal_mask_cast)     
    action_mask_select = torch.cat((torch.zeros(Blan, dtype=torch.bool), torch.ones(Bsub, dtype=torch.bool), torch.zeros(B, dtype=torch.bool), torch.ones(Bbdd, dtype=torch.bool), torch.zeros(Bcast, dtype=torch.bool)), axis=0) #True : use dataset action, False : MBRA action in training code    
    #goal_mask_select = torch.tensor(goal_mask_lan + goal_mask_gnm + goal_mask_frod + goal_mask_bdd)     
    #action_mask_select = torch.cat((torch.zeros(Blan, dtype=torch.bool), torch.ones(Bsub, dtype=torch.bool), torch.zeros(B, dtype=torch.bool), torch.ones(Bbdd, dtype=torch.bool)), axis=0) #True : use dataset action, False : MBRA action in training code
    #goal_mask_select = torch.tensor(goal_mask_frod + goal_mask_gnm + goal_mask_lan)
    merged["goal_mask_select"] = goal_mask_select
    merged["action_mask_select"] = action_mask_select
    #print("merged torch", merged_torch_keys)
    #print("merged list", merged_list_keys)    
    #print("unmerged", unmerged_keys)
    return merged

def merge_batches_padding_small(batch_list, pad_token_id, IGNORE_INDEX, model_max_length):
    """
    Merge a list of dictionary batches into a single dictionary,
    concatenating tensor values along the batch dimension (dim=0).
    Non-tensor values are skipped.
    batch_list is the list of batch data. Order is [lelan, gnm, frod, bdd]
    """
    merged = {}
    keys = batch_list[0].keys()

    #merged_torch_keys = []  
    #merged_list_keys = []        
    #unmerged_keys = []       
    for key in keys:
        #print(key)
        #if key == "input_ids":
        #    print(batch_list[0][key][0].size())
        #    print(batch_list[1][key][0].size())
        values = [batch[key] for batch in batch_list]
        first_value = values[0]
        if key == "temp_dist":
            batch_size  = [batch[key].size()[0] for batch in batch_list]
            batch_tdist = [batch[key] for batch in batch_list]

        if isinstance(first_value, torch.Tensor):
            #merged_torch_keys.append(key)
            #if key == "input_ids":
            #    print("values", values)
            merged[key] = torch.cat(values, dim=0)
        elif isinstance(first_value, list):
            #merged_list_keys.append(key)
            combined_list = []
            for v in values:
                combined_list.extend(v)
            merged[key] = combined_list            
        else:
            #unmerged_keys.append(key)
            # Optionally log or keep one example
            # print(f"Skipping non-tensor key: {key}")
            pass  # or merged[key] = batch_list[0][key]

    input_ids = pad_sequence(merged["input_ids"], batch_first=True, padding_value=pad_token_id)
    merged["input_ids"] = input_ids[:, : model_max_length]
    labels = pad_sequence(merged["labels"], batch_first=True, padding_value=IGNORE_INDEX)
    merged["labels"] = labels[:, : model_max_length]
    merged["attention_mask"] = merged["input_ids"].ne(pad_token_id)  
    merged["attention_mask_label"] = merged["labels"].ne(IGNORE_INDEX)        
    #print(merged["input_ids"].size(), merged["labels"].size(), merged["attention_mask"].size())

    goal_mask_cast = []
    goal_mask_lan = [] 
    Blan = batch_size[0]
    Bcast = batch_size[1]

    distance_lan = batch_tdist[0]
    dist_label_cast = batch_tdist[1]
         
    """
    my_list = [7,8]            
    for idl in range(Blan):
        if batch["temp_dist"][idl] == 0:
            if random.random() > 0.5:
                goal_mask_lan.append(7)
            else:
                goal_mask_lan.append(random.choice(my_list))   
        else:
            goal_mask_lan.append(6)   
    """
    
    #my_list = [4,7,8]
    my_list = [7,8]    
    for idl in range(Blan):
        if distance_lan[idl] == 0:
            if random.random() > 0.5:
                goal_mask_lan.append(random.choice(my_list))   
            else:
                #goal_mask_lan.append(8)   
                goal_mask_lan.append(7)   
        else:
            #goal_mask_lan.append(random.randint(6,7))  
            goal_mask_lan.append(6)                
                                        
    for idg in range(Bcast):
        goal_mask_cast.append(7)
          
    """
    #my_list = [4,7,8]
    #my_list = [7,8]    
    for idl in range(Blan):
        goal_mask_lan.append(6)
                                
    for idf in range(B):
        if distance[idf] <= 20:
            goal_mask_frod.append(random.randint(4,6))
        else:
            goal_mask_frod.append(random.randint(4,5))
                    
    for idg in range(Bsub):
        if dist_label_sub[idg] <= 20:
            goal_mask_gnm.append(random.randint(4,6))
        else:
            goal_mask_gnm.append(random.randint(4,5))
                    
    for idt in range(Bbdd):
        if distance_bdd[idt] <= 20:
            goal_mask_bdd.append(random.randint(4,6))
        else:
            goal_mask_bdd.append(random.randint(4,5))    
    """
    #goal_mask_all = []   
    #for isum in range(Blan + B + Bsub + Bbdd):
    #    goal_mask_all.append(4)
    #    
    # [lelan, gnm, frod, bdd]  
    #goal_mask_select = torch.tensor(goal_mask_all) 
    goal_mask_select = torch.tensor(goal_mask_lan + goal_mask_cast)     
    action_mask_select = torch.cat((torch.zeros(Blan, dtype=torch.bool), torch.zeros(Bcast, dtype=torch.bool)), axis=0) #True : use dataset action, False : MBRA action in training code      
    #goal_mask_select = torch.tensor(goal_mask_lan + goal_mask_gnm)     
    #action_mask_select = torch.cat((torch.zeros(Blan, dtype=torch.bool), torch.ones(Bsub, dtype=torch.bool)), axis=0) #True : use dataset action, False : MBRA action in training code
    #goal_mask_select = torch.tensor(goal_mask_frod + goal_mask_gnm + goal_mask_lan)
    merged["goal_mask_select"] = goal_mask_select
    merged["action_mask_select"] = action_mask_select
    #print("merged torch", merged_torch_keys)
    #print("merged list", merged_list_keys)    
    #print("unmerged", unmerged_keys)
    return merged

class DistributedWeightedSampler(WeightedRandomSampler):
    """
    WeightedRandomSampler that works with DistributedDataParallel (DDP).
    Splits sampled indices evenly across ranks.
    """
    def __init__(self, weights, num_samples, replacement=True, num_replicas=None, rank=None):
        super().__init__(weights, num_samples, replacement)
        if not torch.distributed.is_available():
            raise RuntimeError("Requires torch.distributed")
        if not torch.distributed.is_initialized():
            raise RuntimeError("Requires initialized torch.distributed")

        self.num_replicas = num_replicas or torch.distributed.get_world_size()
        self.rank = rank or torch.distributed.get_rank()

    def __iter__(self):
        # Sample as usual
        indices = list(super().__iter__())

        # Split indices across GPUs
        return iter(indices[self.rank::self.num_replicas])

    def set_epoch(self, epoch: int):
        # for API compatibility with DistributedSampler
        # You could reseed here if you want epoch-wise variation
        self.epoch = epoch

@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    """
    Fine-tunes base VLA on demonstration dataset via LoRA.

    Allows toggling different action representations (discrete vs. continuous), different learning objectives
    (next-token prediction vs. L1 regression vs. diffusion), FiLM. Also allows for additional model inputs,
    such as additional camera images and robot proprioceptive state. Assumes parallel action generation with
    action chunking.

    Args:
        cfg (FinetuneConfig): Training configuration.

    Returns:
        None.
    """
    assert cfg.use_lora, "Only LoRA fine-tuning is supported. Please set --use_lora=True!"
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot do both L1 regression and diffusion. Please pick one of them!"
    )

    # Trim trailing forward slash ('/') in VLA path if it exists
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # Get experiment run ID
    run_id = get_run_id(cfg)

    # Create experiment run directory
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)
    print("run_dir", run_dir)

    with open("./config_nav/defaults.yaml", "r") as f:
        default_conf = yaml.safe_load(f)
    config = default_conf    

    if GPU_server:
        with open("./config_nav/base_server.yaml", "r") as f:        
            user_config = yaml.safe_load(f)        
    else:
        #with open("./config_nav/test.yaml", "r") as f:        
        with open("./config_nav/base.yaml", "r") as f:             
            user_config = yaml.safe_load(f)
    config.update(user_config)
    
    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    #print("device_id", device_id)
    world_size = int(os.environ["WORLD_SIZE"]) 
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    print("World size", world_size, "rank", device_id)

    # Initialize wandb logging
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")
    
    #defining and loading MBRA
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
    if GPU_server:
        load_project_folder_exaug = os.path.join("/home/noriaki//Learning-to-Drive-Anywhere-with-MBRA2/deployment/model_weights")
        latest_path_exaug = os.path.join(load_project_folder_exaug, "exaug_labeler.pth")    
    else:
        load_project_folder_exaug = os.path.join("/home/noriaki/Documents/Learning-to-Drive-Anywhere-with-MBRA_next/deployment/model_weights")
        latest_path_exaug = os.path.join(load_project_folder_exaug, "mbra.pth")
    print("Loading ExAug model from ", load_project_folder_exaug)
    latest_checkpoint_exaug = torch.load(latest_path_exaug, map_location="cpu")
    mbra.load_state_dict(latest_checkpoint_exaug, strict=False) 
    #load_model(mbra, config["model_type"], latest_checkpoint_exaug)
    mbra.eval().to(device=device_id)
    mbra = wrap_ddp(mbra, device_id, find_unused=True)

    # Print detected constants
    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
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
    
    #AutoConfig.register("openvla", OpenVLAConfig)
    #AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    #AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    #AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMN)
    #print("zzzzzz")

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
        #with init_empty_weights():
        #    vla = OpenVLAForActionPrediction_MMN(config_openvla)
    
        vla.load_state_dict(state_dict, strict=False)
        #missing, unexpected = vla.load_state_dict(state_dict, strict=False)
        #print("Missing keys:", missing)
        #print("Unexpected keys:", unexpected)
    
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

    # loading parameters from original ViNT style models
    if GPU_server:
        checkpoint_vint = torch.load("/nfs/kun2/users/noriaki/checkpoints/logs/frodobot-gnm-lelan/frodobot-gnm_2025_05_30_16_54_40/latest_.pth", map_location="cpu")
        state_dict_vint = checkpoint_vint["state_dict"] if "state_dict" in checkpoint_vint else checkpoint_vint    
    else:
        checkpoint_vint = torch.load("/home/noriaki/Documents/Learning-to-Drive-Anywhere-with-MBRA_next/train/logs/frodobot-gnm-lelan/frodobot-gnm_2025_05_30_16_54_40_bdd/latest_.pth", map_location="cpu")
        state_dict_vint = checkpoint_vint["state_dict"] if "state_dict" in checkpoint_vint else checkpoint_vint
    
    """
    my_keys = set(k for k, _ in vla.goal_encoder_img.named_parameters())
    #my_keys_goal_encoder = set(k for k, _ in vla.goal_encoder.named_parameters())
    #my_keys_local_goal = set(k for k, _ in vla.local_goal.named_parameters())
    #my_keys_compress_goal_enc_img = set(k for k, _ in vla.compress_goal_enc_img.named_parameters())    
    #my_keys_compress_obs_enc_map = set(k for k, _ in vla.compress_obs_enc_map.named_parameters())  
    #my_keys = my_keys_goal_encoder_img | my_keys_goal_encoder | my_keys_local_goal | my_keys_compress_goal_enc_img | my_keys_compress_obs_enc_map
    #print("my_keys", my_keys)
    
    filtered_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # Remove any prefix if necessary (e.g., "module.", "model.", etc.)
        stripped_key = key.replace("goal_encoder_img.", "")  # adjust this as needed
        
        if stripped_key in my_keys:
            filtered_state_dict[stripped_key] = value    
    """
    """
    layer_load = ["goal_encoder_img", "goal_encoder", "local_goal", "obs_encoder", "compress_goal_enc_img", "compress_obs_enc_map", "compress_obs_enc"]
    for name in layer_load:
        print("loading ", name)
        module = getattr(vla, name, None)
        my_keys = set(k for k, _ in module.named_parameters())
        filtered_state_dict = OrderedDict()
        for key, value in state_dict.items():
            # Remove any prefix if necessary (e.g., "module.", "model.", etc.)
            stripped_key = key.replace("goal_encoder_img" + ".", "")  # adjust this as needed
        
            if stripped_key in my_keys:
                filtered_state_dict[stripped_key] = value  
        missing, unexpected  = module.load_state_dict(filtered_state_dict, strict=False)
        print("Unexpected keys:", unexpected)
    """      
    #loading parameters
    #missing, unexpected = vla.goal_encoder_img.load_state_dict(filtered_state_dict, strict=False)
    #vla.goal_encoder_img.load_state_dict(filtered_state_dict, strict=False)
    vla.to(dtype=torch.bfloat16, device=device_id)
    #print("Missing keys:", missing)
    #print("Unexpected keys:", unexpected)

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
            #target_modules="all-linear",
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    #freezing pretrained encoders (trial and error)
    """
    for name in layer_load:
        module = getattr(vla, name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False
        else:
            print(f"[Warning] Module {name} not found in the model.")
    vla.print_trainable_parameters()
    """
    # FiLM setup
    print("cfg.use_film", cfg.use_film)
    if cfg.use_film:
        count_parameters(vla.vision_backbone, "vla.vision_backbone (original)")
        # Wrap vision backbone with FiLM wrapper
        # Important: For this, must specify `vla.model.vision_backbone` instead of just `vla.vision_backbone`, since the
        # latter would cause the new wrapped backbone to be saved as a new attribute of `vla` instead of overwriting the
        # original one (due to the LoRA wrapper)
        vla.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla.model.vision_backbone,
            llm_dim=vla.llm_dim,
        )
        count_parameters(vla.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla.model.vision_backbone.load_state_dict(state_dict)
        vla.model.vision_backbone = vla.model.vision_backbone.to(device_id)

    #print(vla)
    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=True)

    # If applicable, instantiate proprio projector
    print("cfg.use_proprio", cfg.use_proprio)
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    # If applicable, instantiate continuous action head for L1 regression
    #print("vla.module.llm_dim", vla.module.llm_dim, "vla.module.llm_dim", vla.module.llm_dim)
    if cfg.use_l1_regression:
        action_head = init_module(
            #L1RegressionActionHead,
            L1RegressionActionHead_idcat,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": ACTION_DIM},
            to_bf16=True,
        )
        dist_head = init_module(
            L1RegressionDistHead,
            "dist_head",
            cfg,
            device_id,
            {"input_dim": vla.module.llm_dim, "hidden_dim": vla.module.llm_dim, "action_dim": 1},
            to_bf16=True,
        )
    #print("vla.module.llm_dim", vla.module.llm_dim, "vla.module.llm_dim", vla.module.llm_dim)


    # If applicable, instantiate diffusion action head and noisy action projector
    if cfg.use_diffusion:
        action_head = init_module(
            DiffusionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_diffusion_steps_train": cfg.num_diffusion_steps_train,
            },
            to_bf16=True,
        )
        noisy_action_projector = init_module(
            NoisyActionProjector, "noisy_action_projector", cfg, device_id, {"llm_dim": vla.module.llm_dim}
        )

    # Get number of vision patches
    print(vla.module.vision_backbone.get_num_patches(), vla.module.vision_backbone.get_num_images_in_input())
    NUM_PATCHES = vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    # If we have proprio inputs, a single proprio embedding is appended to the end of the vision patch embeddings
    if cfg.use_proprio:
        NUM_PATCHES += 1
    # For diffusion, a single diffusion timestep embedding is appended to the end of the vision patch embeddings
    if cfg.use_diffusion:
        NUM_PATCHES += 1
    
    #for multi-modal navigation policy (TODO this will be += 3 or more for adding satellite image and history of images)
    #NUM_PATCHES += 3 + config["context_size"] + 1
    print("NUM_PATCHES", NUM_PATCHES)
    print("cfg.resume", cfg.resume)
    # Instantiate optimizer
    """
    for name, param in vla.named_parameters():
        if "project_img_feat" in name:
           if param.requires_grad:
               print("project_img_feat is learnable parameters:")
    """
    """
    layer_prefixes = ["module.base_model.model.project_img_feat.base_layer", "module.base_model.model.project_pose_feat", "module.base_model.model.project_sate_feat"]
    for name, param in vla.named_parameters():
        if any(name.startswith(prefix) for prefix in layer_prefixes):
            param.requires_grad = True
            print(f"[✓] Unfroze: {name}")    
    """
    if not GPU_server:
        for param in vla.parameters():
            param.requires_grad = False
        #flag = 0
        #for param in action_head.parameters():
        #    if flag == 1:
        #        param.requires_grad = False
        #    flag = 1
                
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.use_l1_regression or cfg.use_diffusion:
        trainable_params += [param for param in action_head.parameters() if param.requires_grad]
        trainable_params += [param for param in dist_head.parameters() if param.requires_grad]
    if cfg.use_diffusion:
        trainable_params += [param for param in noisy_action_projector.parameters() if param.requires_grad]
    if cfg.use_proprio:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]
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
    print("processor.tokenizer", processor.tokenizer)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    #from prismatic.vla.datasets import DummyDataset
    #
    #train_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder,
    #)
    # 
    #visualize lelan dataloader
    #data_iter = iter(train_dataset)
    #for i in range(3):
    #    batch = next(data_iter)
    #    print(batch["pixel_values"].size())    
    # ---

    # We assume that the model takes as input one third-person camera image and 1 or 2 optional wrist camera image(s)
    use_wrist_image = cfg.num_images_in_input > 1
    """
    # Create training and optional validation datasets
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )
    """
    # Create collator and dataloader
    #collator = PaddedCollatorForActionPrediction(
    tokenizer_max_length = processor.tokenizer.model_max_length
    collator = PaddedCollatorForActionPrediction_Nav_MMN(
        tokenizer_max_length, processor.tokenizer.pad_token_id, padding_side="right", num_img = cfg.num_images_in_input
        
    )
    #processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right", num_img = cfg.num_images_in_input    
    if True:
        #with open("./config_nav/base_server.yaml", "r") as f:
        if "lan_solo" not in config:
            config["lan_solo"] = False
        if "sate_solo" not in config:
            config["sate_solo"] = False        
        if "image_solo" not in config:
            config["image_solo"] = False
        if "no_sate" not in config:
            config["no_sate"] = False 
        if "ft_frod" not in config:
            config["ft_frod"] = False                        
        if "aug_shadow" not in config:
            config["aug_shadow"] = False          
        if config["image_solo"] == True:
            config["horizon_long"] = 20
        
        if GPU_server:
            #B_lan = 4
            #B_gnm = 1
            #B_fro = 1
            #B_bdd = 1
            B_lan = 2
            B_cast = 2
            B_gnm = 1
            B_fro = 1
            B_bdd = 1              
            #B_lan = 2
            #B_gnm = 2
            #B_fro = 2
            #B_bdd = 1            
        else:
            B_lan = cfg.batch_size
            B_cast = cfg.batch_size              
            B_gnm = cfg.batch_size
            B_fro = cfg.batch_size
            B_bdd = cfg.batch_size        
        
        train_dataset_lan = []
        test_dataset_lan = []
        for data_split_type in ["train", "test"]:
            #CAST dataset 
            if data_split_type == "train":
                print("Defining CAST training dataset")
                with open(cast_loc + "features.pkl", 'rb') as f:
                    features, num_examples = pickle.load(f)
                                    
                #dataset_name = "cast_filtered_dataset"
                CAST_dataset_list = ["cast_filtered_dataset_convert", "cast_counterfactual_dataset_convert", "atomic_turn_right_dataset_convert", "atomic_turn_left_dataset_convert", "atomic_stop_dataset_convert", "atomic_forward_dataset_convert", "atomic_adjust_right_dataset_convert", "atomic_adjust_left_dataset_convert"]
                CAST_size = [15493, 103125, 27486, 28336, 1293, 94656, 5872, 6706]
                ratios = [0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
                weights = []
                for size, ratio in zip(CAST_size, ratios):
                    weights.extend([ratio / size] * size)
                weights = torch.DoubleTensor(weights)
                
                #CAST_dataset_list = ["cast_filtered_dataset_convert"]
                #CAST_size = [15493]
                train_dataset_CAST_l = []
                for idx, dataset_name in enumerate(CAST_dataset_list):
                    #data_list = np.load(cast_loc + dataset_name + ".npy", allow_pickle=True).tolist()
                    # Load
                    print("CAST dataset name", dataset_name)
    
                    train_dataset_CAST_comp = CAST_Dataset_MMN(action_tokenizer=action_tokenizer,
                        base_tokenizer=processor.tokenizer, 
                        image_transform=processor.image_processor.apply_transform,
                        prompt_builder_fn=PurePromptBuilder,
                        dataset_name=dataset_name,
                        data_loc=cast_loc,
                        data_size=CAST_size[idx],
                        features=features)
                    train_dataset_CAST_l.append(train_dataset_CAST_comp)
                
                train_dataset_CAST = ConcatDataset(train_dataset_CAST_l)
                
                sampler_train_cast = DistributedWeightedSampler(
                    weights, num_samples=len(train_dataset_CAST), replacement=True
                )

                #sampler_train_cast = DistributedSampler(train_dataset_CAST, num_replicas=world_size, rank=device_id, shuffle=True)          
                train_loader_CAST = DataLoader(
                    train_dataset_CAST,
                    batch_size=B_cast,
                    shuffle=False,            
                    num_workers=config["num_workers"],
                    collate_fn=collator,
                    drop_last=True,
                    persistent_workers=True,
                    sampler=sampler_train_cast)          
        
        
            #Frodobots-2k dataset 
            ratio_f = config["ratio_f"]
            split_train_test = int(11994*ratio_f)             
            if data_split_type == "train":
                dataset_Frod = ViNTLeRobotDataset_IL2_gps_map2_crop_shadow_MMN(
                    action_tokenizer=action_tokenizer,
                    base_tokenizer=processor.tokenizer, 
                    image_transform=processor.image_processor.apply_transform,
                    prompt_builder_fn=PurePromptBuilder,                 
                    repo_id=config["repo_id"], 
                    video="video", 
                    root=config["root"], 
                    image_size=config["image_size"], 
                    split="train", 
                    goal_horizon=config["horizon_short"], 
                    goal_horizon2=config["horizon_long"], 
                    sacson=config["SACSoN"], 
                    context_spacing=3, 
                    action_spacing=3)         
                episode_sampler_train = EpisodeSampler_IL_MMN(dataset_Frod, 0, split_train_test, goal_horizon=config["horizon_short"], data_split_type=data_split_type, num_replicas=world_size, rank=device_id)  
                train_loader_Frod = DataLoader(
                    dataset_Frod,
                    batch_size=B_fro,
                    shuffle=False,            
                    num_workers=config["num_workers"],
                    collate_fn=collator,
                    drop_last=True,
                    persistent_workers=True,
                    sampler=episode_sampler_train,
                )                                
            else:
                dataset_Frod = ViNTLeRobotDataset_IL2_gps_map2_crop_shadow_MMN(
                    action_tokenizer=action_tokenizer,
                    base_tokenizer=processor.tokenizer, 
                    image_transform=processor.image_processor.apply_transform,
                    prompt_builder_fn=PurePromptBuilder,                      
                    repo_id=config["repo_id"], 
                    video="video", 
                    root=config["root"], 
                    image_size=config["image_size"], 
                    split="train", 
                    goal_horizon=config["horizon_short"], 
                    goal_horizon2=config["horizon_long"], 
                    sacson=config["SACSoN"], 
                    context_spacing=3, 
                    action_spacing=3)      
                episode_sampler_test = EpisodeSampler_IL_MMN(dataset_Frod, split_train_test, 11994-1, goal_horizon=config["horizon_short"], data_split_type=data_split_type, num_replicas=world_size, rank=device_id)
                test_loader_Frod = DataLoader(
                    dataset_Frod,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=config["num_workers"],
                    collate_fn=collator,
                    drop_last=True,
                    sampler=episode_sampler_test,                
                ) 
            #GNM dataset   
            print("config[goal_type]", config["goal_type"])
            train_dataset_gnm = []
            test_dataset_gnm = [] 
            for dataset_name in config["datasets_sub"]:       
                data_config_sub = config["datasets_sub"][dataset_name]
                if "negative_mining" not in data_config_sub:
                    data_config_sub["negative_mining"] = True
                if "goals_per_obs" not in data_config_sub:
                    data_config_sub["goals_per_obs"] = 1
                if "end_slack" not in data_config_sub:
                    data_config_sub["end_slack"] = 0
                if "waypoint_spacing" not in data_config_sub:
                    data_config_sub["waypoint_spacing"] = 1                        
                if data_split_type in data_config_sub:                   
                    dataset_gnm = ViNT_Dataset_gps_MMN(
                        action_tokenizer=action_tokenizer,
                        base_tokenizer=processor.tokenizer, 
                        image_transform=processor.image_processor.apply_transform,
                        prompt_builder_fn=PurePromptBuilder,                        
                        data_folder=data_config_sub["data_folder"],
                        data_split_folder=data_config_sub[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config_sub["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config_sub["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config_sub["end_slack"],
                        goals_per_obs=data_config_sub["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                    )
                if data_split_type == "train":    
                     train_dataset_gnm.append(dataset_gnm)
                else:    
                     test_dataset_gnm.append(dataset_gnm)
                     
            if data_split_type == "train":                     
                train_dataset_gnm = ConcatDataset(train_dataset_gnm)
                sampler_train_gnm = DistributedSampler(train_dataset_gnm, num_replicas=world_size, rank=device_id, shuffle=True)                    
                train_loader_gnm = DataLoader(
                    train_dataset_gnm,
                    batch_size=B_gnm,
                    shuffle=False,
                    num_workers=config["num_workers"],
                    collate_fn=collator,
                    drop_last=True,
                    persistent_workers=True,
                    sampler=sampler_train_gnm,
                )                  
            else:    
                test_dataset_gnm = ConcatDataset(test_dataset_gnm)
                sampler_test_gnm = DistributedSampler(test_dataset_gnm, num_replicas=world_size, rank=device_id, shuffle=True)  
                test_dataloaders_gmm = DataLoader(
                    test_dataset_gnm,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=config["num_workers"],
                    collate_fn=collator,
                    drop_last=True,
                    persistent_workers=True,                    
                    sampler=sampler_test_gnm,                    
                )      
                
            #BDD dataset     
            data_config_bdd = config["datasets_bdd"]
            dataset_bdd = BDD_Dataset_multi_MMN(
                action_tokenizer=action_tokenizer,
                base_tokenizer=processor.tokenizer, 
                image_transform=processor.image_processor.apply_transform,
                prompt_builder_fn=PurePromptBuilder,              
                data_split_folder=data_config_bdd[data_split_type],
                dataset_name="bdd",
                image_size=config["image_size"],
                waypoint_spacing=data_config_bdd["waypoint_spacing"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                context_size=config["context_size"],
                data_split_type = data_split_type,
                data_folder = data_config_bdd["image"],    
                pickle_folder = data_config_bdd["pickle"],                                                                        
                context_type=config["context_type"],
                normalize=config["normalize"],
                model_type=config["model_type"],
                aug_seq=data_config_bdd["aug_seq"],                                                     
            )   
            if data_split_type == "train":
                sampler_train_bdd = DistributedSampler(dataset_bdd, num_replicas=world_size, rank=device_id, shuffle=True)  
                train_loader_bdd = DataLoader(
                    dataset_bdd,
                    batch_size=B_bdd,
                    shuffle=False,
                    num_workers=config["num_workers"],
                    collate_fn=collator,
                    drop_last=True,
                    persistent_workers=True,
                    sampler=sampler_train_bdd,
                )      
            else:
                sampler_test_bdd = DistributedSampler(dataset_bdd, num_replicas=world_size, rank=device_id, shuffle=True)             
                test_loader_bdd = DataLoader(
                    dataset_bdd,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=config["num_workers"],
                    collate_fn=collator,
                    drop_last=True,
                    persistent_workers=True,
                    sampler=sampler_test_bdd,                    
                )                                         
                                                                      
            #LeLaN dataset                 
            for dataset_name_lan in config["datasets_lan"]:
                data_config_lan = config["datasets_lan"][dataset_name_lan]   
                if "negative_mining" not in data_config_lan:
                    data_config_lan["negative_mining"] = True
                if "goals_per_obs" not in data_config_lan:
                    data_config_lan["goals_per_obs"] = 1
                if "end_slack" not in data_config_lan:
                    data_config_lan["end_slack"] = 0
                if "waypoint_spacing" not in data_config_lan:
                    data_config_lan["waypoint_spacing"] = 1
                                                 
                dataset_lan = LeLaN_Dataset_openvla_act_MMN(
                    action_tokenizer=action_tokenizer,
                    base_tokenizer=processor.tokenizer, 
                    image_transform=processor.image_processor.apply_transform,
                    prompt_builder_fn=PurePromptBuilder,                  
                    data_split_folder=data_config_lan[data_split_type],
                    dataset_name=dataset_name_lan,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config_lan["waypoint_spacing"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    data_split_type = data_split_type,
                    data_image_folder = data_config_lan["image"],
                    data_pickle_folder = data_config_lan["pickle"], 
                    lan_solo = config["lan_solo"],                                                                         
                    context_type=config["context_type"],
                    normalize=config["normalize"],
                    backside=data_config_lan["backside"],
                    aug_seq=data_config_lan["aug_seq"],   
                    only_front=data_config_lan["only_front"],                                                                       
                ) 
                if data_split_type == "train":
                    train_dataset_lan.append(dataset_lan)
                elif data_split_type == "test":
                    test_dataset_lan.append(dataset_lan)
                    
            if data_split_type == "train":                   
                train_dataset_lan = ConcatDataset(train_dataset_lan)

                if True:                
                    dataset_sizes = [len(ds) for ds in train_dataset_lan.datasets]
                    total_size = sum(dataset_sizes)

                    # Weight is inverse of dataset size
                    weights_per_dataset = [1.0 / size for size in dataset_sizes]
                    vis_weights_per_dataset = [w / weights_per_dataset[0] for w in weights_per_dataset]
                    print("Weight for lelan train dataset", dataset_name, weights_per_dataset, vis_weights_per_dataset)
                    # Now map each sample index to its dataset's weight
                    sample_weights = []
                    for size, w in zip(dataset_sizes, weights_per_dataset):
                        sample_weights.extend([w] * size)
                    sample_weights = torch.DoubleTensor(sample_weights)
                
                    sampler_train_lan = WeightedDistributedSampler(
                        train_dataset_lan,
                        weights=sample_weights,
                        num_replicas=world_size,
                        rank=device_id,
                        replacement=True
                    )
                else:
                    sampler_train_lan = DistributedSampler(train_dataset_lan, num_replicas=world_size, rank=device_id, shuffle=True) 
                
                train_loader_lelan = DataLoader(
                    train_dataset_lan,
                    batch_size=B_lan,
                    shuffle=False,
                    collate_fn=collator,
                    num_workers=config["num_workers"],
                    drop_last=True,
                    persistent_workers=True,
                    sampler=sampler_train_lan,
                )                  
            else:
                test_dataset_lan = ConcatDataset(test_dataset_lan) 
                
                if True:
                    dataset_sizes = [len(ds) for ds in train_dataset_lan.datasets]
                    total_size = sum(dataset_sizes)

                    # Weight is inverse of dataset size
                    weights_per_dataset = [1.0 / size for size in dataset_sizes]

                    # Now map each sample index to its dataset's weight
                    sample_weights = []
                    for size, w in zip(dataset_sizes, weights_per_dataset):
                        sample_weights.extend([w] * size)
                    sample_weights = torch.DoubleTensor(sample_weights)
                
                    sampler_test_lan = WeightedDistributedSampler(
                        train_dataset_lan,
                        weights=sample_weights,
                        num_replicas=world_size,
                        rank=device_id,
                        replacement=True
                    )
                else:                
                    sampler_test_lan = DistributedSampler(test_dataset_lan, num_replicas=world_size, rank=device_id, shuffle=True)                 
                val_batch_size = cfg.batch_size
                test_loader_lelan = DataLoader(
                    test_dataset_lan,
                    batch_size=val_batch_size,
                    shuffle=False,
                    collate_fn=collator,
                    num_workers=config["num_workers"],
                    drop_last=True,
                    persistent_workers=True,
                    sampler=sampler_test_lan,
                )                      
            """
            if data_split_type == "train":                   
                train_dataset_lan = ConcatDataset(train_dataset_lan)  
                sampler_train_lan = DistributedSampler(train_dataset_lan, num_replicas=world_size, rank=device_id, shuffle=True) 
                train_loader_lelan = DataLoader(
                    train_dataset_lan,
                    batch_size=B_lan,
                    shuffle=False,
                    collate_fn=collator,
                    num_workers=config["num_workers"],
                    drop_last=True,
                    persistent_workers=True,
                    sampler=sampler_train_lan,
                )                  
            else:
                test_dataset_lan = ConcatDataset(test_dataset_lan) 
                sampler_test_lan = DistributedSampler(test_dataset_lan, num_replicas=world_size, rank=device_id, shuffle=True)                 
                val_batch_size = cfg.batch_size
                test_loader_lelan = DataLoader(
                    test_dataset_lan,
                    batch_size=val_batch_size,
                    shuffle=False,
                    collate_fn=collator,
                    num_workers=config["num_workers"],
                    drop_last=True,
                    persistent_workers=True,
                    sampler=sampler_test_lan,
                )  
            """
                
    #visualize lelan dataloader
    #data_iter = iter(train_loader_lelan)
    #for i in range(10):
    #    batch = next(data_iter)
    #    print(batch)
                                
    # [Important] Save dataset statistics so that we can unnormalize actions during inference
    #print("train_dataset.dataset_statistics", train_dataset.dataset_statistics)
    #if distributed_state.is_main_process:
    #    save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
    """
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )
    """
    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "L1_action_value": deque(maxlen=cfg.grad_accumulation_steps),        
        "L1_obj_value": deque(maxlen=cfg.grad_accumulation_steps),
        "L1_smooth_value": deque(maxlen=cfg.grad_accumulation_steps),   
        "L2_action_value": deque(maxlen=cfg.grad_accumulation_steps),        
        "L2_obj_value": deque(maxlen=cfg.grad_accumulation_steps),
        "L2_smooth_value": deque(maxlen=cfg.grad_accumulation_steps),       
        "L2_dist_value": deque(maxlen=cfg.grad_accumulation_steps),          
        "L2_sate": deque(maxlen=cfg.grad_accumulation_steps),
        "L2_sate_pose": deque(maxlen=cfg.grad_accumulation_steps),        
        "L2_sate_img": deque(maxlen=cfg.grad_accumulation_steps),
        "L2_sate_pose_img": deque(maxlen=cfg.grad_accumulation_steps),   
        "L2_pose": deque(maxlen=cfg.grad_accumulation_steps),        
        "L2_pose_img": deque(maxlen=cfg.grad_accumulation_steps),
        "L2_img": deque(maxlen=cfg.grad_accumulation_steps),       
        "L2_lan": deque(maxlen=cfg.grad_accumulation_steps),          
        "L2_lan_pose": deque(maxlen=cfg.grad_accumulation_steps),                                            
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }

    # Start training
    lelan_iter = iter(train_loader_lelan)
    gnm_iter = iter(train_loader_gnm)
    frod_iter = iter(train_loader_Frod)
    bdd_iter = iter(train_loader_bdd) 
    cast_iter = iter(train_loader_CAST)        
    #iters = [gnm_iter, gnm_iter, frod_iter, frod_iter]
    
    if GPU_server:    
        #iters = [lelan_iter, gnm_iter, frod_iter, bdd_iter]
        #samplers = [sampler_train_lan, sampler_train_gnm, episode_sampler_train, sampler_train_bdd]
        iters = [lelan_iter, gnm_iter, frod_iter, bdd_iter, cast_iter]
        samplers = [sampler_train_lan, sampler_train_gnm, episode_sampler_train, sampler_train_bdd, sampler_train_cast]        
    else:
        #iters = [lelan_iter, gnm_iter]
        #samplers = [sampler_train_lan, sampler_train_gnm]
        iters = [lelan_iter, lelan_iter]
        samplers = [sampler_train_lan, sampler_train_lan]              
        #iters = [lelan_iter, cast_iter]
        #samplers = [sampler_train_lan, sampler_train_cast]          
    log_count = 0
    for epoch in range(100):
        for sampler in samplers:
            sampler.set_epoch(epoch)
                
        with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
            if GPU_server:
                print("setting up training mode")
                vla.train()
            else:
                print("setting up eval (Local PC coding) mode")
                vla.eval()
                action_head.eval()
                dist_head.eval()
                proprio_projector.eval()
                
            optimizer.zero_grad()
            #for batch_idx, batch in enumerate(dataloader):
            #for batch_idx, batch in enumerate(train_loader_lelan):
            for batch_idx in range(cfg.max_steps):
                batches = []
                for i, it in enumerate(iters):
                    try:
                        batch = next(it)
                    except StopIteration:
                        iters[i] = iter([train_loader_lelan, train_loader_gnm, train_loader_Frod, train_loader_bdd, train_loader_CAST][i])
                        #iters[i] = iter([train_loader_lelan, train_loader_gnm, train_loader_Frod, train_loader_bdd][i])
                        batch = next(iters[i])
                    batches.append(batch)
                #merged_batch = merge_batches(batches)    
                #, , padding_side="right"
                #print("processor.tokenizer.model_max_length", processor.tokenizer.model_max_length)
                if GPU_server: 
                    merged_batch = merge_batches_padding(batches, processor.tokenizer.pad_token_id, IGNORE_INDEX, tokenizer_max_length)
                else:
                    merged_batch = merge_batches_padding_small(batches, processor.tokenizer.pad_token_id, IGNORE_INDEX, tokenizer_max_length)                    
                #merged_batch = merge_batches_padding(batches, processor.tokenizer.pad_token_id, IGNORE_INDEX, processor.tokenizer.model_max_length)
                """            
                try:
                    batch_gnm = next(gnm_iter)
                except StopIteration:
                    gnm_iter = iter(train_loader_gnm) 
                    batch_gnm = next(gnm_iter)                 

                try:
                    batch_frod = next(frod_iter)
                except StopIteration:
                    frod_iter = iter(train_loader_Frod) 
                    batch_frod = next(frod_iter)                 

                try:
                    batch_bdd = next(bdd_iter)
                except StopIteration:
                    bdd_iter = iter(train_loader_bdd) 
                    batch_bdd = next(bdd_iter)   
                """    
                #print(batch_bdd)
                #print("batch[goal_pose]", merged_batch["goal_pose"])
                """    
                print("pixel_values", batch["pixel_values"].size(), batch_gnm["pixel_values"].size(), batch_frod["pixel_values"].size(), batch_bdd["pixel_values"].size())
                print("actions", batch["actions"].size(), batch_gnm["actions"].size(), batch_frod["actions"].size(), batch_bdd["actions"].size())
                print("goal_pose", batch["goal_pose"].size(), batch_gnm["goal_pose"].size(), batch_frod["goal_pose"].size(), batch_bdd["goal_pose"].size())
                print("obj_pose_norm", batch["obj_pose_norm"].size(), batch_gnm["obj_pose_norm"].size(), batch_frod["obj_pose_norm"].size(), batch_bdd["obj_pose_norm"].size())
                print("cur_image_crop", batch["cur_image_crop"].size(), batch_gnm["cur_image_crop"].size(), batch_frod["cur_image_crop"].size(), batch_bdd["cur_image_crop"].size())
                print("cur_image", batch["cur_image"].size(), batch_gnm["cur_image"].size(), batch_frod["cur_image"].size(), batch_bdd["cur_image"].size())
                print("goal_image_crop", batch["goal_image_crop"].size(), batch_gnm["goal_image_crop"].size(), batch_frod["goal_image_crop"].size(), batch_bdd["goal_image_crop"].size())
                print("goal_image_8", batch["goal_image_8"].size(), batch_gnm["goal_image_8"].size(), batch_frod["goal_image_8"].size(), batch_bdd["goal_image_8"].size())                
                print("temp_dist", batch["temp_dist"].size(), batch_gnm["temp_dist"].size(), batch_frod["temp_dist"].size(), batch_bdd["temp_dist"].size())     
                print("input_ids", batch["input_ids"].size(), batch_gnm["input_ids"].size(), batch_frod["input_ids"].size(), batch_bdd["input_ids"].size())   
                print("labels", batch["labels"].size(), batch_gnm["labels"].size(), batch_frod["labels"].size(), batch_bdd["labels"].size())      
                print("attention_mask", batch["attention_mask"].size(), batch_gnm["attention_mask"].size(), batch_frod["attention_mask"].size(), batch_bdd["attention_mask"].size())                                                                       
                """    
                # Compute training metrics and loss
                #print("In the loop", batch["pose_obj"].size(), batch["pose_obj"])
                compute_diffusion_l1 = cfg.use_diffusion and batch_idx % cfg.diffusion_sample_freq == 0
                loss, metrics = run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    dist_head=dist_head,
                    mbra=mbra,
                    noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                    proprio_projector=proprio_projector if cfg.use_proprio else None,
                    batch=merged_batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    use_l1_regression=cfg.use_l1_regression,
                    use_diffusion=cfg.use_diffusion,
                    use_proprio=cfg.use_proprio,
                    use_film=cfg.use_film,
                    num_patches=NUM_PATCHES,
                    compute_diffusion_l1=compute_diffusion_l1,
                    num_diffusion_steps_train=cfg.num_diffusion_steps_train if cfg.use_diffusion else None,
                    mode="train",
                    #mode="vali",
                    idrun=batch_idx,
                )
                #batch = next(data_iter)            
                # Normalize loss to account for gradient accumulation
                normalized_loss = loss / cfg.grad_accumulation_steps
                        
                # Backward pass
                if GPU_server:
                    normalized_loss.backward()
                #else:
                #    normalized_loss.backward()

                """
                for name, param in vla.named_parameters():
                    if "project_img_feat" in name:
                        if param.grad is not None:
                            print(f"[✓] Gradient found for {name}, grad norm: {param.grad.norm().item():.4f}")
                        else:
                            print(f"[✗] No gradient for {name}")
                """
                """
                for name, param in action_head.named_parameters():
                    if param.grad is None:
                        print(f"{name}: ⚠️ No gradient")
                    else:
                        print(f"{name}: ✅ gradient")                    
                """
                # Store recent train metrics
                for metric_name, value in metrics.items():
                    if metric_name in recent_metrics:
                        recent_metrics[metric_name].append(value)

                # Compute gradient step index
                #gradient_step_idx = (batch_idx + epoch*len(train_loader_lelan)) // cfg.grad_accumulation_steps
                gradient_step_idx = log_count // cfg.grad_accumulation_steps
                log_count += 1
                # Compute smoothened train metrics
                smoothened_metrics = compute_smoothened_metrics(recent_metrics)

                # Push Metrics to W&B (every wandb_log_freq gradient steps)
                log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            
            
                #print(distributed_state.is_main_process, log_step % cfg.wandb_log_freq, gradient_step_idx % cfg.wandb_log_freq)
                #print("cfg.wandb_log_freq", cfg.wandb_log_freq, "gradient_step_idx", gradient_step_idx, "log_step", log_step)
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
                    #print("save wandb")
                    wandb.log(
                        {
                            "VLA Train/Learning Rate": scheduler.get_last_lr()[0],
                        },
                        step=log_step,
                    )

                #old_weights_vlm = [param.clone() for param in vla.parameters()]
                #old_weights_act = [param.clone() for param in action_head.parameters()]
                #old_weights_dist = [param.clone() for param in dist_head.parameters()]

                # Optimizer and LR scheduler step
                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    progress.update()
                """
                new_weights_vlm = [param for param in vla.parameters()]
                new_weights_act = [param for param in action_head.parameters()] 
                new_weights_dist = [param for param in dist_head.parameters()]                                                
                for i, (old, new) in enumerate(zip(old_weights_vlm, new_weights_vlm)):
                    print(f"Layer {i} weight change VLM: {(old - new).abs().mean()}")
                for i, (old, new) in enumerate(zip(old_weights_act, new_weights_act)):
                    print(f"Layer {i} weight change ACT: {(old - new).abs().mean()}")
                for i, (old, new) in enumerate(zip(old_weights_dist, new_weights_dist)):
                    print(f"Layer {i} weight change DIST: {(old - new).abs().mean()}")  
                """
                # Save model checkpoint: either keep latest checkpoint only or all checkpoints
                if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                #if True:            
                    save_training_checkpoint(
                        cfg=cfg,
                        run_dir=run_dir,
                        log_step=log_step,
                        vla=vla,
                        processor=processor,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                        action_head=action_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                        dist_head=dist_head if (cfg.use_l1_regression or cfg.use_diffusion) else None,
                        train_dataset=train_dataset_lan,
                        distributed_state=distributed_state,
                    )

                # Test model on validation set
                #if cfg.use_val_set and log_step > 0 and log_step % cfg.val_freq == 0:
                if False:
                    print("eval on validation process")
                    run_validation(
                        vla=vla,
                        action_head=action_head,
                        dist_head=dist_head,
                        mbra=mbra,
                        noisy_action_projector=noisy_action_projector if cfg.use_diffusion else None,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        #val_dataloader=val_dataloader,
                        val_dataloader=test_loader_lelan,
                        action_tokenizer=action_tokenizer,
                        device_id=device_id,
                        cfg=cfg,
                        num_patches=NUM_PATCHES,
                        log_step=log_step,
                        distributed_state=distributed_state,
                        val_time_limit=cfg.val_time_limit,
                    )
                    # Set model back to training mode after validation
                    vla.train()

                ## Stop training when max_steps is reached
                #if log_step == cfg.max_steps:
                #    print(f"Max step {cfg.max_steps} reached! Stopping training...")
                #    break

if __name__ == "__main__":
    finetune()
