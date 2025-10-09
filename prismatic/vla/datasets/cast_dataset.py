#import sys
#sys.path.append('/media/noriaki/Noriaki_Data/Learning-to-Drive-Anywhere-with-MBRA/train/')

import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import torchvision.transforms.functional as TF
import numpy as np
import pickle
import random
from PIL import Image

from typing import Any, Dict, List, Optional, Tuple, Type
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform

from vint_train.data.data_utils import (
    img_path_to_data,
    img_path_to_data_depth,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

class CAST_Dataset_MMN(Dataset):
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
        """
        Args:
            data_list: List of examples, each example is a dict containing:
                - 'observation': dict with keys like 'image', 'state', etc.
                - 'action': np.array
                - 'language_instruction': str
        """
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

    def _compute_actions(self, action_yaw, goal_pose, metric_waypoint):#traj_data, curr_time, goal_time
        #start_index = curr_time
        #end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        #yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
        #positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]
        #goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]
        #goal_yaw = traj_data["yaw"][min(goal_time, len(traj_data["position"]) - 1)]

        positions = action_yaw[:,0:2]
        yaw = action_yaw[:,2]

        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pose[:,0:2], positions[0], yaw[0])
        
        #print(goal_pos, goal_pose[:,0:2])
        #goal_yaw_loc = goal_yaw - yaw[0]
        
        #assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        if True:
            yaw = yaw[1:] - yaw[0]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
            yawg = goal_pose[:,2:3] - yaw[0]
            #print(goal_pos.shape, yawg.shape)
            goal_pos = np.concatenate([goal_pos, yawg], axis=1)
        else:
            actions = waypoints[1:]
            goal_pos = goal_pos
        if True:
            actions[:, :2] /= metric_waypoint
            goal_pos[:, :2] /= metric_waypoint
        #print(goal_pos)
        #    goal_pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
        #
        #assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"
        # 
        #return actions, goal_pos, goal_yaw_loc
        return torch.from_numpy(actions), torch.from_numpy(goal_pos)

    def __getitem__(self, idx):
        #path = self.data_list[0]
        #dataset = tf.data.TFRecordDataset([path]).map(self.features.deserialize_example)
        #subsample = 1
        #traj = next(iter(dataset.take(idx)))["steps"].batch(int(1e9)).get_single_element()
        #traj = tf.nest.map_structure(lambda x: x.numpy(), traj)
        #episode_metadata = next(iter(dataset.take(idx)))["episode_metadata"]
        
        
        folder_name = self.dataset_name.split("_convert")
        directory_location = self.data_loc + self.dataset_name + "/" + folder_name[0] + "/"
        
        len_action = 0
        while len_action < 10:
            traj = np.load(directory_location + f"traj_{idx:06d}.npz", allow_pickle=True)
            len_action = len(traj['action'])
            if len_action < 10:
                idx = random.randint(0, self.data_size-1)
            #print(folder_name, len_action)
            
        
        #len(traj['action']) - 8
        num = random.randint(0, len(traj['action']) - 8 - 2)
        gid = max(len(traj['action']) - 1, num + 8)
        
        obs_dict = traj["observation"].item() 
        #print(traj['observation']['image'].shape)
        cur_pilimg = obs_dict['image'][num]
        goal_pilimg = obs_dict['image'][gid]
        
        cur_obs = cur_pilimg.transpose(2, 0, 1)
        goal_obs = goal_pilimg.transpose(2, 0, 1)

        # convert to PIL
        pil_img = Image.fromarray(cur_pilimg.astype(np.uint8)).resize(self.image_size_clip) 
        pil_img_goal = Image.fromarray(goal_pilimg.astype(np.uint8)).resize(self.image_size_clip) 

        pixel_values = self.image_transform(pil_img)
        pixel_values_g = self.image_transform(pil_img_goal)
        
        action_yaw = obs_dict['state'][num: num+8+1]
        goal_pose = obs_dict['state'][gid:gid+1]
        #action_yaw = traj['action_angle'][num: num+8+1]   
        #goal_pose = traj['action_angle'][gid:gid+1]          
        
        #print("action_yaw", action_yaw)
        actions_norm, goal_pose_norm = self._compute_actions(action_yaw, goal_pose, traj["normalization_factor"])
        #print("actions_norm", actions_norm)        
        actions_torch = calculate_sin_cos(actions_norm) #[x, y, cos, sin]
        goal_pose_torch = calculate_sin_cos(goal_pose_norm) #[x, y, cos, sin]

        #select from the prompt list          
        language_instruction = traj['language_instruction'][0]
        non_empty_prompts = [p for p in language_instruction if p]
        selected_prompt = random.choice(non_empty_prompts).decode('utf-8')
        #print(selected_prompt)
        """
        return {
            'observation': cur_obs,
            'action': actions_torch,
            'language_instruction': selected_prompt
        }
        """
        ### Adapting OpenVLA stle ###
        actions = actions_torch
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)        
                
        try:
            lang = selected_prompt.lower()
        except:
            print(inst_obj_x) 
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            #{"from": "human", "value": f"No language instruction"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder("openvla")
        
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        
        max_token = 60
        if len(input_ids) > max_token: 
            try:
                lang = "move toward " + "XXXXX"
            except:
                print(inst_obj_x) 
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                #{"from": "human", "value": f"No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
            # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
            prompt_builder = self.prompt_builder("openvla")
            
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            # Tokenize (w/ `base_tokenizer`)
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
            labels = list(input_ids)    
        
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)   
        
        obj_pose_norm = goal_pose_torch[0, 0:2]
        goal_pose_cos_sin = goal_pose_torch.squeeze(0)
        
        current_map_image = np.asarray(np.random.rand(3, 96, 96), dtype=np.float32)
        goal_map_image = np.asarray(np.random.rand(3, 96, 96), dtype=np.float32)
        dummy_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        dummy_PIL = Image.fromarray(dummy_array)
        pixel_values_dummy = self.image_transform(dummy_PIL)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
                    
        goal_id = 0
        cur_image_r = self._resize_norm(torch.from_numpy(cur_obs), self.image_size).repeat(6,1,1)/255.0
        goal_image_full_r = self._resize_norm(torch.from_numpy(goal_obs), self.image_size)/255.0
        goal_image_full_8_r = self._resize_norm(torch.from_numpy(goal_obs), self.image_size)/255.0

        dataset_name = "cast"
        #print(selected_prompt)
                        
        return_dict = dict(pixel_values=pixel_values, pixel_values_wrist=pixel_values_g, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions_torch, actions_nomad=actions_torch, goal_pose=goal_pose_cos_sin, obj_pose_norm=obj_pose_norm, img_PIL=pil_img, cur_image_crop = cur_image_r, cur_image = cur_image_r, cur_image_large = cur_image_r, goal_image_crop=goal_image_full_r, goal_image_8=goal_image_full_8_r, temp_dist=goal_id, cur_map_image=current_map_image, goal_map_image=goal_map_image, pixel_values_curmap=pixel_values_dummy, pixel_values_goalmap=pixel_values_dummy)
        
        return return_dict
        

def collate_fn(batch):
    """
    Custom collate_fn to handle batching variable-length sequences if needed.
    """
    obs_batch = {}
    # Collate observation dicts
    for key in batch[0]['observation']:
        obs_batch[key] = torch.stack([ex['observation'][key] for ex in batch])

    actions = torch.stack([ex['action'] for ex in batch])
    instructions = [ex['language_instruction'] for ex in batch]

    return {
        'observation': obs_batch,
        'action': actions,
        'language_instruction': instructions
    }


# Usage
if __name__ == "__main__":
    # Suppose data_list is a list of your processed examples
    dataset_name = "cast_filtered_dataset"
    data_list = np.load(dataset_name + ".npy", allow_pickle=True).tolist()
    # Load
    with open('features.pkl', 'rb') as f:
        features, num_examples = pickle.load(f)
    print(features)
    
    dataset = CAST_Dataset_MMN(data_list, features, num_examples)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch in dataloader:
        print(batch['observation'].shape)  # e.g., [8, H, W, C]
        print(batch['action'].shape)                 # e.g., [8, action_dim]
        print(batch['language_instruction'])         # list of 8 strings
        break

