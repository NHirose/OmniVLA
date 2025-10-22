import os
import io
import json
import pickle
import random
import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import lmdb
import yaml
import utm
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image

from prismatic.vla.constants import IGNORE_INDEX
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform


def img_path_to_data_front(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """Load an image and convert to tensor."""
    return TF.to_tensor(Image.open(path))


class BDD_Dataset(Dataset):
    def __init__(
        self,
        action_tokenizer: PreTrainedTokenizerBase,
        base_tokenizer: ActionTokenizer,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        data_split_type: str,
        data_folder: str,
        pickle_folder: str,
        predict_stop_token: bool = True,
        context_type: str = "temporal",
        normalize: bool = True,
        aug_seq: bool = False,
    ):
        self.data_split_folder = data_split_folder
        self.data_split_type = data_split_type
        self.data_folder = data_folder
        self.pickle_folder = pickle_folder
        self.image_size = image_size
        self.image_size_clip = (224, 224)
        self.waypoint_spacing = waypoint_spacing
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.context_size = context_size
        self.context_type = context_type
        self.normalize = normalize
        self.aug_seq = aug_seq
        self.dataset_name = dataset_name

        self.num_action_params = 3 if learn_angle else 2

        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.prompt_builder = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.image_transform = image_transform

        # Load dataset config
        with open(os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r") as f:
            all_data_config = yaml.safe_load(f)
        assert self.dataset_name in all_data_config, f"Dataset {self.dataset_name} not found"
        dataset_names = sorted(all_data_config.keys())
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]

        self.trajectory_cache = {}
        self._load_split_index()
        self._build_caches_front()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches_front()

    def _build_caches_front(self, use_tqdm: bool = True):
        """Build LMDB cache for faster image loading."""
        cache_filename = os.path.join(
            self.data_split_folder, f"dataset_{self.dataset_name}_{self.data_split_type}.lmdb"
        )

        self._get_mbra_traj()

        if not os.path.exists(cache_filename):
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for num in range(len(self.image_path)):
                        for ih in range(self.context_size):
                            with open(self.image_path[num][ih], "rb") as f:
                                txn.put(self.image_path[num][ih].encode(), f.read())

        self._image_cache_path = cache_filename
        self._image_cache = None

    def _sample_negative(self):
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _remove_values_from_list(self, A, B):
        return [item for item in A if item not in B]

    def _load_split_index(self):
        if self.dataset_name == "bdd":
            self.v_random = 0.2
            self.h_random = 0.1
            image_path, gps_list, pickle_path = [], [], []

            folder_lst = next(os.walk(self.data_folder))[1]

            if self.data_split_type == "train":
                folder_lst_dataset = folder_lst[: int(len(folder_lst) * 0.9)]
            else:
                folder_lst_dataset = folder_lst[int(len(folder_lst) * 0.9) :]

            for folder in tqdm(folder_lst_dataset):
                with open(os.path.join(self.data_folder, folder, "gps.json"), "r") as f:
                    gps_data = json.load(f)
                gps_list += gps_data
                number_files = len(gps_data)
                for num in range(number_files):
                    image_path_hist = [
                        os.path.join(self.data_folder, folder, "img", f"{max(0, gps_data[num]['img_id']-ih)}.jpg")
                        for ih in range(self.context_size)
                    ]
                    image_path.append(image_path_hist)
                    pickle_path.append(os.path.join(self.pickle_folder, folder, "pickle", f"{gps_data[num]['img_id']}.pkl"))

        self.image_path = image_path
        self.pickle_path = pickle_path
        self.gps_list = gps_list

    def _get_mbra_traj(self):
        mbra_traj_list = []
        for num in tqdm(range(len(self.pickle_path)), desc="BDD pickle loading"):
            if os.path.getsize(self.pickle_path[num]) > 0:
                with open(self.pickle_path[num], "rb") as f:
                    aug_data = pickle.load(f)
            else:
                print(self.pickle_path[num])
                aug_data = np.zeros((1, self.num_action_params))
            mbra_traj_list.append(aug_data)
        self.mbra_traj_list = mbra_traj_list

    def _get_image_cache(self):
        if self._image_cache is None:
            self._image_cache = lmdb.open(
                self._image_cache_path, readonly=True, lock=False, readahead=False, max_readers=2048
            )
        return self._image_cache

    def _load_image_front(self, path):
        try:
            env = self._get_image_cache()
            with env.begin() as txn:
                image_bytes = io.BytesIO(txn.get(path.encode()))
            return img_path_to_data_front(image_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load image {path}")

    def _resize_norm(self, image, size):
        return TF.resize(image, size)

    def _calculate_relative_position(self, x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    def _rotate_to_local_frame(self, delta_x, delta_y, heading_a_rad):
        relative_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        relative_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return relative_x, relative_y

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        flag_data = 0
        iv = i
        
        it = random.random()
        while flag_data == 0:
            image_fullsize = self._load_image_front(self.image_path[iv][0])
            PIL_current = to_pil_image(image_fullsize).resize((224, 224))
            
            context_image = [image_fullsize]        
            for ih in range(self.context_size):
                context_image.append(self._load_image_front(self.image_path[iv][ih]))             

            #goal_id = random.randint(0,5)
            goal_id = 1
            cur_pos = self.gps_list[iv]    
            mbra_traj = self.mbra_traj_list[iv]                        
            try:
                goal_image_full = self._load_image_front(self.image_path[iv + goal_id][0])
                goal_image_full_8 = self._load_image_front(self.image_path[iv + 1][0])     
                PIL_goal = to_pil_image(goal_image_full).resize((224, 224))
                
                curr_path = self.image_path[iv][0]   
                goal_path = self.image_path[iv + 1][0]           
                goal_pos = self.gps_list[iv + goal_id]
            except:
                goal_id = 0
                goal_image_full = self._load_image_front(self.image_path[iv + goal_id][0])   
                goal_image_full_8 = goal_image_full
                PIL_goal = to_pil_image(goal_image_full).resize((224, 224))
                goal_pos = self.gps_list[iv + goal_id]     
                curr_path = self.image_path[iv][0]   
                goal_path = self.image_path[iv + goal_id][0]   
                            
            try:                
                cur_compass = -cur_pos["course"]/180.0*3.141592
                goal_compass = -goal_pos["course"]/180.0*3.141592
            except:
                cur_compass = 0.0/180.0*3.141592
                goal_compass = 0.0/180.0*3.141592
                                
            cur_utm = utm.from_latlon(cur_pos["latitude"], cur_pos["longitude"])
            goal_utm = utm.from_latlon(goal_pos["latitude"], goal_pos["longitude"])
            delta_x, delta_y = self._calculate_relative_position(cur_utm[0], cur_utm[1], goal_utm[0], goal_utm[1])
            relative_x, relative_y = self._rotate_to_local_frame(delta_x, delta_y, cur_compass)
            radius = np.sqrt(relative_x**2 + relative_y**2)
            
            if it > 0.5 and abs(goal_compass - cur_compass) < 3.141592*20.0/180.0:
                iv = random.randint(0, len(self.image_path)-1)
            else:
                flag_data = 1
            
            curr_path_list = curr_path.split("/")
            goal_path_list = goal_path.split("/")            
            limit_radius = 3.0    
            if radius > 15.0 or curr_path_list[-3] != goal_path_list[-3]:
                iv = random.randint(0, len(self.image_path)-1)
                flag_data = 0
            else: 
                if radius > limit_radius and abs(goal_compass - cur_compass) < 3.141592*30.0/180.0:              
                    relative_x = relative_x/radius*limit_radius
                    relative_y = relative_y/radius*limit_radius

        voffset = int(224.0*self.v_random*random.random())
        hoffset = int(224.0*self.h_random*random.random())

        image_obs_list = [] 
        image_crop_list = [] 
        metric_waypoint_spacing = 0.25 #normalization 
        for ih in range(self.context_size + 1):
            image_crop_list.append(self._resize_norm(context_image[ih][:, voffset:224-voffset, hoffset:224-hoffset], self.image_size))  
            image_obs_list.append(self._resize_norm(context_image[ih], self.image_size))          
        goal_image_full_8 = self._resize_norm(goal_image_full_8, self.image_size) 
        cur_image_large = self._resize_norm(context_image[0][:, voffset:224-voffset, hoffset:224-hoffset], self.image_size_clip)                 

        PILbox = (hoffset, voffset, 224-hoffset, 224-voffset)
        PIL_current_crop = PIL_current.crop(PILbox).resize(self.image_size_clip) 
        PIL_goal_crop = PIL_goal.crop(PILbox).resize(self.image_size_clip) 
                                          
        image_obs = torch.cat(image_obs_list[::-1])      
        image_crop = torch.cat(image_crop_list[::-1])                
       
        if random.random() > 0.5:
            image_obs_r = torch.flip(image_obs, [2])
            image_crop_r = torch.flip(image_crop, [2])
            cur_image_large_r = torch.flip(cur_image_large, [2])
            goal_image_full_8_r = torch.flip(goal_image_full_8, [2])            
            goal_pose_r = np.array([relative_y/metric_waypoint_spacing, relative_x/metric_waypoint_spacing, np.cos(-goal_compass+cur_compass), np.sin(-goal_compass+cur_compass)])      

            PIL_current_crop_r = PIL_current_crop.transpose(Image.FLIP_LEFT_RIGHT)            
            PIL_goal_crop_r = PIL_goal_crop.transpose(Image.FLIP_LEFT_RIGHT)             
            mbra_traj[:,1] = -mbra_traj[:,1]
            mbra_traj[:,3] = -mbra_traj[:,3]                  
        else:
            image_obs_r = image_obs
            image_crop_r = image_crop
            cur_image_large_r = cur_image_large
            goal_image_full_8_r = goal_image_full_8            
            goal_pose_r = np.array([relative_y/metric_waypoint_spacing, -relative_x/metric_waypoint_spacing, np.cos(goal_compass-cur_compass), np.sin(goal_compass-cur_compass)])    
            
            PIL_current_crop_r = PIL_current_crop
            PIL_goal_crop_r = PIL_goal_crop                             
        
        goad_id_virtual = -1
        for ig in range(20):
            if 0.3333*ig < radius and radius < 0.3333*(ig + 1):
                goad_id_virtual = ig
        if goad_id_virtual == -1:
            goad_id_virtual = 20
        if goal_id > 1:
            goad_id_virtual = 100

        obj_pose_norm = np.array((0.0, 0.0)) #dummy obj pose
        goal_pose_cos_sin = goal_pose_r #Adapting ViNT style action commands (X, Y, cos, sin)            
         
        ### Adapting OpenVLA stle ###
        actions = mbra_traj
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"No language instruction"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder("openvla")
        
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        
        pixel_values = self.image_transform(PIL_current_crop_r)
        pixel_values_g = self.image_transform(PIL_goal_crop_r)
                
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        dataset_name = "bdd"
        
        # Set the available modality id for each dataset 
        # 0:"satellite only", 1:"pose and satellite", 2:"satellite and image", 3:"all", 4:"pose only", 5:"pose and image", 6:"image only", 7:"language only", 8:"language and pose"        
        modality_list = [4, 5, 6]   
        if goad_id_virtual <= 20:
            modality_id = random.choice(modality_list)
        else:
            modality_id = random.choice(modality_list[0:2]) #tdisntace is long --> no image only

        # action select 1.0: BDD fintuned MBRA synthetic action, 0.0: MBRA synthetic action       
        # While action_select_mask = 1 for BDD dataset, we do not use the raw action commands.        
        # For BDD, we discard entire action commands due to the large emobidiment gap and generate the action commands with MBRA.
        # We re-trained the another MBRA model with BDD dataset. Different from the other datasets, we previously saved their action commands in pickle files and load them in the dataloader.  
        action_select_mask = torch.tensor(1.0)           
                    
        return dict(
            pixel_values=pixel_values, 
            pixel_values_goal=pixel_values_g, 
            input_ids=input_ids, 
            labels=labels, 
            dataset_name=dataset_name, 
            modality_id=modality_id,
            actions=torch.as_tensor(actions), 
            action_select_mask = action_select_mask,
            goal_pose=goal_pose_cos_sin, 
            obj_pose_norm=obj_pose_norm, 
            img_PIL=PIL_current_crop_r,
            gimg_PIL=PIL_goal_crop_r,
            cur_image = image_obs_r, 
            goal_image_8=goal_image_full_8_r, 
            temp_dist=goad_id_virtual,
            lan_prompt="No language instruction"            
        )                     
