# ===============================================================
# OmniVLA Inference
# ===============================================================
# 
# Sample inference code for OmniVLA
# if you want to control the robot, you need to update the current state such as pose and image in "run_omnivla" and comment out "break" in "run".
#
# ---------------------------
# Paths and System Setup
# ---------------------------
import sys, os
sys.path.insert(0, '..')

import time, math, json
from typing import Optional, Tuple, Type, Dict
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import utm

# ---------------------------
# Custom Imports
# ---------------------------
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat, L1RegressionDistHead
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE

from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor

# ===============================================================
# Utility Functions
# ===============================================================
def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt")) and module_name == "pose_projector":
        module_name = "proprio_projector"
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)

def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")

def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: "InferenceConfig",
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
) -> DDP:
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return module

# ===============================================================
# Inference Class
# ===============================================================
class Inference:
    def __init__(self, save_dir, lan_inst_prompt, goal_utm, goal_compass, goal_image_PIL, action_tokenizer, processor):
        self.tick_rate = 3
        self.lan_inst_prompt = lan_inst_prompt
        self.goal_utm = goal_utm
        self.goal_compass = goal_compass
        self.goal_image_PIL = goal_image_PIL
        self.action_tokenizer = action_tokenizer
        self.processor = processor
        self.count_id = 0
        self.linear, self.angular = 0.0, 0.0
        self.datastore_path_image = save_dir
    # ----------------------------
    # Static Utility Methods
    # ----------------------------
    @staticmethod
    def calculate_relative_position(x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    @staticmethod
    def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    # ----------------------------
    # Main Loop
    # ----------------------------
    def run(self):
        loop_time = 1 / self.tick_rate
        start_time = time.time()
        while True:
            if time.time() - start_time > loop_time:
                self.tick()
                start_time = time.time()
                break

    def tick(self):
        self.linear, self.angular = self.run_omnivla()

    # ----------------------------
    # OmniVLA Inference
    # ----------------------------
    def run_omnivla(self):
        thres_dist = 30.0
        metric_waypoint_spacing = 0.1

        # Load current GPS & heading
        current_lat = 37.87371258374039
        current_lon = -122.26729417226024
        current_compass = 270.0
        cur_utm = utm.from_latlon(current_lat, current_lon)
        cur_compass = -float(current_compass) / 180.0 * math.pi  # inverted compass

        # Local goal position
        delta_x, delta_y = self.calculate_relative_position(
            cur_utm[0], cur_utm[1], self.goal_utm[0], self.goal_utm[1]
        )
        relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
        radius = np.sqrt(relative_x**2 + relative_y**2)
        if radius > thres_dist:
            relative_x *= thres_dist / radius
            relative_y *= thres_dist / radius

        goal_pose_loc_norm = np.array([
            relative_y / metric_waypoint_spacing,
            -relative_x / metric_waypoint_spacing,
            np.cos(self.goal_compass - cur_compass),
            np.sin(self.goal_compass - cur_compass)
        ])

        # Load current image
        current_image_path = "./inference/current_img.jpg"
        current_image_PIL = Image.open(current_image_path).convert("RGB")

        # Language instruction
        lan_inst = self.lan_inst_prompt if lan_prompt else "xxxx"

        # Prepare batch
        batch = self.data_transformer_omnivla(
            current_image_PIL, lan_inst, self.goal_image_PIL, goal_pose_loc_norm,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor
        )

        # Run forward pass
        actions, modality_id = self.run_forward_pass(
            vla=vla.eval(),
            action_head=action_head.eval(),
            noisy_action_projector=None,
            pose_projector=pose_projector.eval(),
            batch=batch,
            action_tokenizer=self.action_tokenizer,
            device_id=device_id,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_patches=NUM_PATCHES,
            compute_diffusion_l1=False,
            num_diffusion_steps_train=None,
            mode="train",
            idrun=self.count_id,
        )
        self.count_id += 1

        waypoints = actions.float().cpu().numpy()

        # Select waypoint
        waypoint_select = 4
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint

        # PD controller
        EPS = 1e-8
        DT = 1 / 3
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * clip_angle(np.arctan2(hy, hx)) / DT
        elif np.abs(dx) < EPS:
            linear_vel_value = 0
            angular_vel_value = 1.0 * np.sign(dy) * np.pi / (2 * DT)
        else:
            linear_vel_value = dx / DT
            angular_vel_value = np.arctan(dy / dx) / DT

        linear_vel_value = np.clip(linear_vel_value, 0, 0.5)
        angular_vel_value = np.clip(angular_vel_value, -1.0, 1.0)

        # Velocity limitation
        maxv, maxw = 0.3, 0.3
        if np.abs(linear_vel_value) <= maxv:
            if np.abs(angular_vel_value) <= maxw:
                linear_vel_value_limit = linear_vel_value
                angular_vel_value_limit = angular_vel_value
            else:
                rd = linear_vel_value / angular_vel_value
                linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                angular_vel_value_limit = maxw * np.sign(angular_vel_value)
        else:
            if np.abs(angular_vel_value) <= 0.001:
                linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                angular_vel_value_limit = 0.0
            else:
                rd = linear_vel_value / angular_vel_value
                if np.abs(rd) >= maxv / maxw:
                    linear_vel_value_limit = maxv * np.sign(linear_vel_value)
                    angular_vel_value_limit = maxv * np.sign(angular_vel_value) / np.abs(rd)
                else:
                    linear_vel_value_limit = maxw * np.sign(linear_vel_value) * np.abs(rd)
                    angular_vel_value_limit = maxw * np.sign(angular_vel_value)

        # Save behavior
        self.save_robot_behavior(
            current_image_PIL, self.goal_image_PIL, goal_pose_loc_norm, waypoints[0],
            linear_vel_value_limit, angular_vel_value_limit, metric_waypoint_spacing, modality_id.cpu().numpy()
        )

        print("linear angular", linear_vel_value_limit, angular_vel_value_limit)
        return linear_vel_value_limit, angular_vel_value_limit

    # ----------------------------
    # Save Robot Behavior Visualization
    # ----------------------------
    def save_robot_behavior(self, cur_img, goal_img, goal_pose, waypoints,
                            linear_vel, angular_vel, metric_waypoint_spacing, mask_number):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs = fig.add_gridspec(2, 2)
        ax_ob = fig.add_subplot(gs[0, 0])
        ax_goal = fig.add_subplot(gs[1, 0])
        ax_graph_pos = fig.add_subplot(gs[:, 1])

        ax_ob.imshow(np.array(cur_img).astype(np.uint8))
        ax_goal.imshow(np.array(goal_img).astype(np.uint8))

        x_seq = waypoints[:, 0] #generated trajectory is on the robot coordinate. X is front and Y is left. 
        y_seq_inv = -waypoints[:, 1]           
        ax_graph_pos.plot(np.insert(y_seq_inv, 0, 0.0), np.insert(x_seq, 0, 0.0), linewidth=4.0, markersize=12, marker='o', color='blue')

        # Mask annotation
        mask_type = int(mask_number[0])
        mask_texts = [
            "satellite only", "pose and satellite", "satellite and image", "all",
            "pose only", "pose and image", "image only", "language only", "language and pose"
        ]
        if mask_type < len(mask_texts):
            ax_graph_pos.annotate(mask_texts[mask_type], xy=(1.0, 0.0), xytext=(-20, 20), fontsize=18, textcoords='offset points')

        ax_ob.set_title("Egocentric current image", fontsize=18)
        ax_goal.set_title("Egocentric goal image", fontsize=18)
        ax_graph_pos.tick_params(axis='x', labelsize=15) 
        ax_graph_pos.tick_params(axis='y', labelsize=15) 
        
        if int(mask_number[0]) == 1 or int(mask_number[0]) == 3 or int(mask_number[0]) == 4 or int(mask_number[0]) == 5 or int(mask_number[0]) == 8:
            ax_graph_pos.plot(-goal_pose[1], goal_pose[0], marker = '*', color='red', markersize=15)  
        else:                           
            ax_graph_pos.set_xlim(-3.0, 3.0)
            ax_graph_pos.set_ylim(-0.1, 10.0)
        ax_graph_pos.set_xlim(-3.0, 3.0)
        ax_graph_pos.set_ylim(-0.1, 10.0)
                        
        ax_graph_pos.set_title("Normalized generated 2D trajectories from OmniVLA", fontsize=18)
        
        save_path = os.path.join(self.datastore_path_image, f"{self.count_id}_ex.jpg")
        plt.savefig(save_path)

    # ----------------------------
    # Custom Collator
    # ----------------------------
    def collator_custom(self, instances, model_max_length, pad_token_id, padding_side="right", pixel_values_dtype=torch.float32):
        IGNORE_INDEX = -100
        input_ids = pad_sequence([inst["input_ids"] for inst in instances], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([inst["labels"] for inst in instances], batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, :model_max_length], labels[:, :model_max_length]
        attention_mask = input_ids.ne(pad_token_id)

        pixel_values = [inst["pixel_values_current"] for inst in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [inst["dataset_name"] for inst in instances]
        else:
            dataset_names = None

        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_goal" in instances[0]:
                pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_goal)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type: {type(pixel_values)}")

        actions = torch.stack([torch.from_numpy(np.copy(inst["actions"])) for inst in instances])
        goal_pose = torch.stack([torch.from_numpy(np.copy(inst["goal_pose"])) for inst in instances])

        output = dict(
            pixel_values=pixel_values.to(),
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            goal_pose=goal_pose,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

    # ----------------------------
    # Transform Data to Dataset Format
    # ----------------------------
    def transform_datatype(self, inst_obj, actions, goal_pose_cos_sin,
                           current_image_PIL, goal_image_PIL, prompt_builder, action_tokenizer,
                           base_tokenizer, image_transform, predict_stop_token=True):
        IGNORE_INDEX = -100
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        if inst_obj == "xxxx":
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {inst_obj}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]

        prompt_builder = prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = torch.tensor(base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX

        pixel_values_current = image_transform(current_image_PIL)
        pixel_values_goal = image_transform(goal_image_PIL)
        dataset_name = "lelan"

        return dict(
            pixel_values_current=pixel_values_current,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            actions=torch.as_tensor(actions),
            goal_pose=goal_pose_cos_sin,
            img_PIL=current_image_PIL,
            inst=inst_obj,
        )

    # ----------------------------
    # Data Transformer for OmniVLA
    # ----------------------------
    def data_transformer_omnivla(self, current_image_PIL, lan_inst, goal_image_PIL, goal_pose_loc_norm,
                                 prompt_builder, action_tokenizer, processor):
        actions = np.random.rand(8, 4)  # dummy actions
        goal_pose_cos_sin = goal_pose_loc_norm

        batch_data = self.transform_datatype(
            lan_inst, actions, goal_pose_cos_sin,
            current_image_PIL, goal_image_PIL,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
        )

        batch = self.collator_custom(
            instances=[batch_data],
            model_max_length=processor.tokenizer.model_max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            padding_side="right"
        )
        return batch

    # ----------------------------
    # Run Forward Pass
    # ----------------------------
    def run_forward_pass(self, vla, action_head, noisy_action_projector, pose_projector,
                         batch, action_tokenizer, device_id, use_l1_regression, use_diffusion,
                         use_film, num_patches, compute_diffusion_l1=False,
                         num_diffusion_steps_train=None, mode="vali", idrun=0) -> Tuple[torch.Tensor, Dict[str, float]]:

        metrics = {}
        noise, noisy_actions, diffusion_timestep_embeddings = None, None, None

        # Determine modality
        if satellite and not lan_prompt and not pose_goal and not image_goal:
            modality_id = torch.as_tensor([0], dtype=torch.float32)
        elif satellite and not lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([1], dtype=torch.float32)
        elif satellite and not lan_prompt and not pose_goal and image_goal:
            modality_id = torch.as_tensor([2], dtype=torch.float32)
        elif satellite and not lan_prompt and pose_goal and image_goal:
            modality_id = torch.as_tensor([3], dtype=torch.float32)
        elif not satellite and not lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([4], dtype=torch.float32)
        elif not satellite and not lan_prompt and pose_goal and image_goal:
            modality_id = torch.as_tensor([5], dtype=torch.float32)
        elif not satellite and not lan_prompt and not pose_goal and image_goal:
            modality_id = torch.as_tensor([6], dtype=torch.float32)
        elif not satellite and lan_prompt and not pose_goal and not image_goal:
            modality_id = torch.as_tensor([7], dtype=torch.float32)
        elif not satellite and lan_prompt and pose_goal and not image_goal:
            modality_id = torch.as_tensor([8], dtype=torch.float32)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                modality_id=modality_id.to(torch.bfloat16).to(device_id),
                labels=batch["labels"].to(device_id),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id),
                proprio_projector=pose_projector,
                noisy_actions=noisy_actions if use_diffusion else None,
                noisy_action_projector=noisy_action_projector if use_diffusion else None,
                diffusion_timestep_embeddings=diffusion_timestep_embeddings if use_diffusion else None,
                use_film=use_film,
            )

        # Prepare data for metrics
        ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
         
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

        with torch.no_grad():
            predicted_actions = action_head.predict_action(actions_hidden_states, modality_id.to(torch.bfloat16).to(device_id))                                 

        # Return both the loss tensor (with gradients) and the metrics dictionary (with detached values)
        return predicted_actions, modality_id

                
# ===============================================================
# Inference Configuration
# ===============================================================
class InferenceConfig:
    resume: bool = True
    vla_path: str = "./omnivla-original"
    resume_step: Optional[int] = 120000    
    #vla_path: str = "./omnivla-finetuned-cast"    
    #resume_step: Optional[int] = 210000
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

def define_model(cfg: InferenceConfig) -> None:
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Loading OpenVLA Model `{cfg.vla_path}`")

    # GPU setup
    device_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPOSE_DIM: {POSE_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
    
    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device_id) #            trust_remote_code=True,
    
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.to(dtype=torch.bfloat16, device=device_id)
    
    pose_projector = init_module(
        ProprioProjector,
        "pose_projector",
        cfg,
        device_id,
        {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM},            
    )
    
    if cfg.use_l1_regression:
        action_head = init_module(
            L1RegressionActionHead_idcat,
            "action_head",
            cfg,
            device_id,
            {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM},            
            to_bf16=True,
        )            
 
    # Get number of vision patches
    NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()    
    NUM_PATCHES += 1 #for goal pose

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    return vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor

# ===============================================================
# Main Entry
# ===============================================================
if __name__ == "__main__":
    # select modality
    pose_goal = False
    satellite = False
    image_goal = True
    lan_prompt = False

    # Goal definitions
    lan_inst_prompt = "move toward blue trash bin"
    goal_lat, goal_lon, goal_compass = 37.8738930785863, -122.26746181032362, 0.0
    goal_utm = utm.from_latlon(goal_lat, goal_lon)
    goal_compass = -float(goal_compass) / 180.0 * math.pi
    goal_image_PIL = Image.open("./inference/goal_img.jpg").convert("RGB")

    # Define models (VLA, action_head, pose_projector, processor, etc.)
    cfg = InferenceConfig()
    vla, action_head, pose_projector, device_id, NUM_PATCHES, action_tokenizer, processor = define_model(cfg)

    # Run inference
    inference = Inference(
        save_dir="./inference",
        lan_inst_prompt=lan_inst_prompt,
        goal_utm=goal_utm,
        goal_compass=goal_compass,
        goal_image_PIL=goal_image_PIL,
        action_tokenizer=action_tokenizer,
        processor=processor,
    )
    inference.run()
