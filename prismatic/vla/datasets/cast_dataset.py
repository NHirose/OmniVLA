import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np
import random
from PIL import Image
from typing import Type
from prismatic.vla.constants import IGNORE_INDEX
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform
from vint_train.data.data_utils import calculate_sin_cos, to_local_coords


class CAST_Dataset(Dataset):
    def __init__(
        self, 
        action_tokenizer: PreTrainedTokenizerBase,
        base_tokenizer: ActionTokenizer, 
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        dataset_name,
        data_loc,
        data_size, 
        features,
        predict_stop_token: bool = True,
    ):
        self.dataset_name = dataset_name
        self.data_loc = data_loc
        self.data_size = data_size        
        self.features = features
        self.image_size = (96, 96)
        self.image_size_clip = (224, 224)

        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.prompt_builder = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.image_transform = image_transform
        
    def __len__(self):
        return self.data_size

    def _resize_norm(self, image, size):
        return TF.resize(image, size)

    def _compute_actions(self, action_yaw, goal_pose, metric_waypoint):
        positions = action_yaw[:, 0:2]
        yaw = action_yaw[:, 2]

        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pose[:, 0:2], positions[0], yaw[0])
        
        yaw = yaw[1:] - yaw[0]
        actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        yawg = goal_pose[:, 2:3] - yaw[0]
        goal_pos = np.concatenate([goal_pos, yawg], axis=1)

        actions[:, :2] /= metric_waypoint
        goal_pos[:, :2] /= metric_waypoint
        
        return torch.from_numpy(actions), torch.from_numpy(goal_pos)

    def __getitem__(self, idx):
        folder_name = self.dataset_name.split("_convert")
        directory_location = self.data_loc + self.dataset_name + "/" + folder_name[0] + "/"
        
        len_action = 0
        while len_action < 10:
            traj = np.load(directory_location + f"traj_{idx:06d}.npz", allow_pickle=True)
            len_action = len(traj['action'])
            if len_action < 10:
                idx = random.randint(0, self.data_size - 1)
            
        num = random.randint(0, len(traj['action']) - 8 - 2)
        gid = max(len(traj['action']) - 1, num + 8)
        
        obs_dict = traj["observation"].item() 
        cur_pilimg = obs_dict['image'][num]
        goal_pilimg = obs_dict['image'][gid]
        
        cur_obs = cur_pilimg.transpose(2, 0, 1)
        goal_obs = goal_pilimg.transpose(2, 0, 1)

        pil_img = Image.fromarray(cur_pilimg.astype(np.uint8)).resize(self.image_size_clip) 
        pil_img_goal = Image.fromarray(goal_pilimg.astype(np.uint8)).resize(self.image_size_clip) 

        pixel_values = self.image_transform(pil_img)
        pixel_values_g = self.image_transform(pil_img_goal)
        
        action_yaw = obs_dict['state'][num: num+8+1]
        goal_pose = obs_dict['state'][gid:gid+1]
        
        actions_norm, goal_pose_norm = self._compute_actions(
            action_yaw, goal_pose, traj["normalization_factor"]
        )       
        actions_torch = calculate_sin_cos(actions_norm)
        goal_pose_torch = calculate_sin_cos(goal_pose_norm)

        language_instruction = traj['language_instruction'][0]
        non_empty_prompts = [p for p in language_instruction if p]
        selected_prompt = random.choice(non_empty_prompts).decode('utf-8')

        actions = actions_torch
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)        
                
        lang = selected_prompt.lower()
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]

        prompt_builder = self.prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        
        max_token = 60
        if len(input_ids) > max_token: 
            lang = "move toward XXXXX"
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]
            prompt_builder = self.prompt_builder("openvla")
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            labels = list(input_ids)    
        
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)   
        
        obj_pose_norm = goal_pose_torch[0, 0:2]
        goal_pose_cos_sin = goal_pose_torch.squeeze(0)
        
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
                    
        goal_id = 0
        cur_image_r = self._resize_norm(torch.from_numpy(cur_obs), self.image_size).repeat(6,1,1)/255.0
        goal_image_full_8_r = self._resize_norm(torch.from_numpy(goal_obs), self.image_size)/255.0

        dataset_name = "cast"
        modality_id = 7  # language only
        action_select_mask = torch.tensor(1.0)           
                    
        return dict(
            pixel_values=pixel_values, 
            pixel_values_goal=pixel_values_g, 
            input_ids=input_ids, 
            labels=labels, 
            dataset_name=dataset_name, 
            modality_id=modality_id,
            actions=torch.as_tensor(actions_torch), 
            action_select_mask=action_select_mask,
            goal_pose=goal_pose_cos_sin, 
            obj_pose_norm=obj_pose_norm, 
            img_PIL=pil_img,
            gimg_PIL=pil_img_goal,
            cur_image=cur_image_r, 
            goal_image_8=goal_image_full_8_r, 
            temp_dist=goal_id,
            lan_prompt=lang       
        )

