#import sys
#sys.path.append('/media/noriaki/Noriaki_Data/Learning-to-Drive-Anywhere-with-MBRA/train/')
#sys.path.append('/home/noriaki/Learning-to-Drive-Anywhere-with-MBRA2/train/')

import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple, Type
import tqdm
import io
import lmdb
import utm
import math

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import random
#import cv2
import matplotlib.pyplot as plt

from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, STOP_INDEX
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform
from PIL import Image
from typing import Union

from vint_train.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

class Dummy_Dataset(Dataset):
    def __init__(
        self,
        context_size: int,
        action_tokenizer: PreTrainedTokenizerBase,
        base_tokenizer: ActionTokenizer,   
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],        
        predict_stop_token: bool = True,            
    ):
        self.context_size = context_size
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.prompt_builder = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.image_transform = image_transform

    def __len__(self) -> int:
        return 10000 #dummy length

    def calculate_relative_position(self, x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    def rotate_to_local_frame(self, delta_x, delta_y, heading_a_rad):
        rel_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    def _resize_norm(self, image, size):
        return TF.resize(image, size)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:   
        thres_dist = 30.0
        metric_waypoint_spacing = 0.1    
        predict_stop_token=True
                   
        # set the available modality id for each dataset 
        # 0:"satellite only", 1:"pose and satellite", 2:"satellite and image", 3:"all", 4:"pose only", 5:"pose and image", 6:"image only", 7:"language only", 8:"language and pose"   
        modality_list = [4, 6, 7]  #[4, 5, 6, 7, 8] # Our sample data is no consistency between modalities. So we can take a solo modality.
        modality_id = random.choice(modality_list)
        
        inst_obj = "move toward blue trash bin"
        actions = np.random.rand(8, 4) #dummy action
        
        # Dummy current and goal location
        current_lat, current_lon, current_compass = 37.87371258374039, -122.26729417226024, 270.0
        cur_utm = utm.from_latlon(current_lat, current_lon)
        cur_compass = -float(current_compass) / 180.0 * math.pi  # inverted compass        
        
        goal_lat, goal_lon, goal_compass = 37.8738930785863, -122.26746181032362, 0.0
        goal_utm = utm.from_latlon(goal_lat, goal_lon)
        goal_compass = -float(goal_compass) / 180.0 * math.pi
        
        # Local goal position
        delta_x, delta_y = self.calculate_relative_position(
            cur_utm[0], cur_utm[1], goal_utm[0], goal_utm[1]
        )
        relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
        radius = np.sqrt(relative_x**2 + relative_y**2)
        if radius > thres_dist:
            relative_x *= thres_dist / radius
            relative_y *= thres_dist / radius

        goal_pose_loc_norm = np.array([
            relative_y / metric_waypoint_spacing,
            -relative_x / metric_waypoint_spacing,
            np.cos(goal_compass - cur_compass),
            np.sin(goal_compass - cur_compass)
        ])        

        goal_pose_cos_sin = goal_pose_loc_norm             
        current_image_PIL = Image.open("./inference/current_img.jpg").convert("RGB")
        goal_image_PIL = Image.open("./inference/goal_img.jpg").convert("RGB")        
                           
        IGNORE_INDEX = -100
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        if modality_id != 7 and modality_id != 8: # We give following language prompt when not selecting language modality instead of masking out the tokens.
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {inst_obj}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]

        prompt_builder = self.prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = torch.tensor(self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # Images for MBRA model
        image_obs_list = []
        for ih in range(self.context_size + 1):
            image_obs_list.append(self._resize_norm(TF.to_tensor(current_image_PIL), (96, 96)))  #In our real code, image_obs_list is list of history image. In this dummy dataset code, we feed current images. The detail implementation is same as ViNT, NoMaD code base. 
        image_obs = torch.cat(image_obs_list)     
        image_goal = self._resize_norm(TF.to_tensor(goal_image_PIL), (96, 96))

        # Data augmentation (random cropping)
        voffset = int(224.0*0.2*random.random())
        hoffset = int(224.0*0.1*random.random())            
        PILbox = (hoffset, voffset, 224-hoffset, 224-voffset)
        current_image_PIL = current_image_PIL.crop(PILbox).resize((224,224)) 
        goal_image_PIL = goal_image_PIL.crop(PILbox).resize((224,224))      

        # Data augmentation (horizontal flipping)
        if random.random() > 0.5:
            current_image_PIL = current_image_PIL.transpose(Image.FLIP_LEFT_RIGHT)
            goal_image_PIL = goal_image_PIL.transpose(Image.FLIP_LEFT_RIGHT)
            actions[:,1] = -actions[:,1]
            actions[:,3] = -actions[:,3]
            goal_pose_cos_sin[1] = -goal_pose_cos_sin[1]
            goal_pose_cos_sin[3] = -goal_pose_cos_sin[3]  
            
            image_obs = torch.flip(image_obs, dims=[2])
            image_goal = torch.flip(image_goal, dims=[2])  
                           
        pixel_values_current = self.image_transform(current_image_PIL)
        pixel_values_goal = self.image_transform(goal_image_PIL)   
        
        #action select 1.0: raw action, 0.0: MBRA synthetic action
        action_select_mask = torch.tensor(1.0)
         
        dataset_name = "dummy"
        return dict(
            pixel_values=pixel_values_current,      #Current image for OmniVLA
            pixel_values_goal=pixel_values_goal,    #Goal image for OmniVLA
            input_ids=input_ids,                    #language and action prompt, following OpenVLA-OFT              
            labels=labels,                          #language and action prompt, following OpenVLA-OFT
            dataset_name=dataset_name,              #dataset name
            modality_id=modality_id,                #modality ID, 0:"satellite only", 1:"pose and satellite", 2:"satellite and image", 3:"all", 4:"pose only", 5:"pose and image", 6:"image only", 7:"language only", 8:"language and pose"
            actions=torch.as_tensor(actions),       #action commands
            action_select_mask = action_select_mask,#action select mask, 1.0: raw action, 0.0: MBRA synthetic action
            goal_pose=goal_pose_cos_sin,            #goal pose [X, Y, cos(yaw), sin(yaw)]
            obj_pose_norm=goal_pose_cos_sin[0:2],   #obj pose [X, Y] (This is only for LeLaN dataset) : Dummy pose in this dummy dataset      
            img_PIL=current_image_PIL,              #for visualization
            gimg_PIL=goal_image_PIL,                #for visualization       
            cur_image=image_obs,                    #History of image for MBRA
            goal_image_8=image_goal,                #Goal image (8 step future) for MBRA
            temp_dist=10.0,                         #Temporal distance (We are not using in our training)
            lan_prompt=inst_obj            
        )      

