import os
import io
import pickle
import yaml
import random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

from prismatic.vla.constants import IGNORE_INDEX
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform


def img_path_to_data_front(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """Load image and convert to tensor."""
    return TF.to_tensor(Image.open(path))


def img_path_to_data_front_PIL(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> Image.Image:
    """Load image and return PIL.Image."""
    return Image.open(path)


class LeLaN_Dataset(Dataset):
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
        data_image_folder: str,
        data_pickle_folder: str,
        predict_stop_token: bool = True,
        context_type: str = "temporal",
        normalize: bool = True,
        backside: bool = False,
        aug_seq: bool = False,
        only_front: bool = False,
    ):
        """Main LeLaN Dataset class."""

        # General config
        self.data_split_folder = data_split_folder
        self.data_split_type = data_split_type
        self.data_image_folder = data_image_folder
        self.data_pickle_folder = data_pickle_folder
        self.image_size = image_size
        self.image_size_clip = (224, 224)
        self.waypoint_spacing = waypoint_spacing
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.context_size = context_size
        assert context_type in {"temporal", "randomized", "randomized_temporal"}, \
            "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.normalize = normalize
        self.backside = backside
        self.aug_seq = aug_seq
        self.dataset_name = dataset_name
        self.only_front = only_front

        # Load dataset configuration
        with open(os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r") as f:
            all_data_config = yaml.safe_load(f)
        assert self.dataset_name in all_data_config, f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = sorted(all_data_config.keys())
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]

        # Tokenizers and prompt builder
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.prompt_builder = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.image_transform = image_transform

        # Action parameters
        self.num_action_params = 3 if self.learn_angle else 2

        # Load dataset indices
        self._load_split_index()
        self._get_augdata()
        self._build_caches_front()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches_front()

    # ----------------------------------------
    # Dataset loading / caching
    # ----------------------------------------
    def _load_split_index(self):
        if self.dataset_name == "go_stanford4":
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping
            
            lst = os.listdir(self.data_image_folder + "image/") # your directory path
            number_files = len(lst)
            
            image_path = []
            pickle_path = []  
            
            ratio = 0.9
            thres = int(number_files*ratio)        
             
            #TODO -5 is come from "self.data_image_folder" includes 5 files, which is not pickle file.
            for num in range(int(number_files - 5)-3):
                if self.data_split_type == "train" and num < thres:
                    image_path.append(self.data_image_folder + "image/" + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + "pickle_nomad/" + str(num).zfill(8) + '.pkl')                               
                elif self.data_split_type == "test" and num >= thres:
                    image_path.append(self.data_image_folder + "image/" + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + "pickle_nomad/" + str(num).zfill(8) + '.pkl')                                     
      
        if self.dataset_name == "sacson":
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping 
                        
            image_path = []
            pickle_path = []   
                                     
            folder_lst = next(os.walk(self.data_pickle_folder))[1]

            if self.data_split_type == "train":
                folder_lst_dataset = folder_lst[0:len(folder_lst)-1]
            else:
                folder_lst_dataset = folder_lst[len(folder_lst)-1:len(folder_lst)]
                                
            for folder in folder_lst_dataset:
                subfolder_lst = os.listdir(self.data_pickle_folder + folder + "/")                                    
                for subfolder in subfolder_lst:
                    file_lst = os.listdir(self.data_image_folder + folder + "/" + subfolder + "/image/")
                    number_files = len(file_lst)
                    for num in range(int(number_files)-3):
                        image_path.append(self.data_image_folder + folder + "/" + subfolder + "/image/" + str(num).zfill(8) + '.jpg')
                        pickle_path.append(self.data_pickle_folder + folder + "/" + subfolder + "/pickle_nomad/" + str(num).zfill(8) + '.pkl')                        

        if self.dataset_name == "go_stanford2":
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping 
                        
            image_path = []
            pickle_path = [] 
                                     
            folder_lst = next(os.walk(self.data_pickle_folder))[1]
            num_test = int(0.1*len(folder_lst))
            
            if self.data_split_type == "train":
                folder_lst_dataset = folder_lst[0:len(folder_lst)-num_test]
            else:
                folder_lst_dataset = folder_lst[len(folder_lst)-num_test:len(folder_lst)]
            
            for folder in folder_lst_dataset:
                file_lst = os.listdir(self.data_image_folder + folder + "/image/")
                number_files = len(file_lst)
                for num in range(int(number_files-3)):
                    image_path.append(self.data_image_folder + folder + "/image/" + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + folder + "/pickle_nomad/" + str(num).zfill(8) + '.pkl')                               

        if self.dataset_name == "humanw":
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping 
                    
            image_path = []
            pickle_path = []                                     
            folder_lst = next(os.walk(self.data_pickle_folder))[1]            
            test_folder = ["R0010096", "R0010098","R0010121", "R0010118","R0010133", "R0010145", "R0010156", "R0010166", "R0010175","R0010180", "R0010188", "R0010197"]
            
            if self.data_split_type == "train":
                folder_lst_dataset = self._remove_values_from_list(folder_lst, test_folder)
            else:
                folder_lst_dataset = test_folder
            
            for folder in folder_lst_dataset:
                file_lst = os.listdir(self.data_image_folder + folder + "/image/")
                number_files = len(file_lst)
                for num in range(int(number_files)):
                    image_path.append(self.data_image_folder + folder + "/image/" + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + folder + "/pickle_nomad/" + str(num).zfill(8) + '.pkl')                         

        if self.dataset_name == "youtube":
            self.v_random = 0.05 #for random cropping
            self.h_random = 0.05 #for random cropping 
                    
            image_path = []
            pickle_path = []                                       
            folder_lst = next(os.walk(self.data_pickle_folder))[1]
            test_folder = ["home_10", "home_12", "austra_1", "spain_1", "singa_1", "spain_3", "spain_5", "rosia_2", "home_33", "poland_1", "uk_5"]
              
            if self.data_split_type == "train":
                folder_lst_dataset = self._remove_values_from_list(folder_lst, test_folder)          
            else:
                folder_lst_dataset = test_folder  
            
            for folder in folder_lst_dataset:
                file_lst = os.listdir(self.data_image_folder + folder + "/image/")
                number_files = len(file_lst)
                for num in range(int(number_files)):
                    image_path.append(self.data_image_folder + folder + "/image/" + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + folder + "/pickle_nomad/" + str(num).zfill(8) + '.pkl')                        
          
        self.image_path = image_path
        self.pickle_path = pickle_path

    def _get_augdata(self):
        """Load all pickle data into memory."""
        self.aug_data_list = []
        for path in self.pickle_path:
            if os.path.getsize(path) > 0:
                with open(path, "rb") as f:
                    aug_data = pickle.load(f)
            else:
                print(f"Empty pickle: {path}")
                aug_data = []
            self.aug_data_list.append(aug_data)

    def _build_caches_front(self):
        """Build LMDB cache for faster image loading."""
        cache_file = os.path.join(self.data_split_folder, f"dataset_{self.dataset_name}_{self.data_split_type}.lmdb")
        if not os.path.exists(cache_file):
            with lmdb.open(cache_file, map_size=2**40) as env:
                with env.begin(write=True) as txn:
                    for img_path in self.image_path:
                        with open(img_path, "rb") as f:
                            txn.put(img_path.encode(), f.read())
        self._image_cache_path = cache_file
        self._image_cache = None

    def _get_image_cache(self):
        if self._image_cache is None:
            self._image_cache = lmdb.open(
                self._image_cache_path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=2048
            )
        return self._image_cache

    def _load_image_front(self, path):
        """Load image from LMDB."""
        try:
            env = self._get_image_cache()
            with env.begin() as txn:
                buffer = txn.get(path.encode())
            return img_path_to_data_front(io.BytesIO(buffer), self.image_size)
        except TypeError:
            print(f"Failed to load image {path}")
            return None

    def _load_image_front_PIL(self, path):
        """Load image as PIL from LMDB."""
        try:
            env = self._get_image_cache()
            with env.begin() as txn:
                buffer = txn.get(path.encode())
            return img_path_to_data_front_PIL(io.BytesIO(buffer), self.image_size)
        except TypeError:
            print(f"Failed to load image {path}")
            return None

    # ----------------------------------------
    # Helper functions
    # ----------------------------------------
    def _resize_norm(self, image, size):
        return TF.resize(image, size)

    def _sample_negative(self):
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _remove_values_from_list(self, A, B):
        return [item for item in A if item not in B]

    # ----------------------------------------
    # Dataset API
    # ----------------------------------------
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        flag_data = 0
        iv = i

        while flag_data == 0:
            image_fullsize_PIL = self._load_image_front_PIL(self.image_path[iv])                            
            image_fullsize = self._load_image_front(self.image_path[iv])                        
            context_image = [image_fullsize]        
            for ih in range(self.context_size):
                if iv-ih > 0:                   
                    context_image.append(self._load_image_front(self.image_path[iv-ih]))             
                else:
                    context_image.append(self._load_image_front(self.image_path[0]))          
            
            for ih in range(self.context_size + 1):
                if context_image[ih] is None:
                    iv = random.randint(0, len(self.image_path)-1)
                            
            if random.random() > 0.8:
                goal_id = random.randint(0,20)
            else:
                goal_id = 0                

            try:
                goal_image_full_8 = self._load_image_front(self.image_path[iv + 8])
            except:
                goal_image_full_8 = self._load_image_front(self.image_path[iv])  
    
            try:
                gimage_fullsize_PIL = self._load_image_front_PIL(self.image_path[iv + goal_id])              
            except:
                goal_id = 0
                gimage_fullsize_PIL = self._load_image_front_PIL(self.image_path[iv + goal_id])
                
            pickle_values = self.aug_data_list[iv + goal_id]                          
            if len(pickle_values) != 0:
                list_rand = [random.randint(0, len(pickle_values)-1) for i in range(len(pickle_values))]  
                il = 0                
                ir = list_rand[il]        
                c_pose_check = 0
                
                pose_obj = pickle_values[ir]["pose_median"][0] #pose on robot coordinate
                
                try:                           
                    nomad_traj_norm = pickle_values[ir]["nomad_traj_norm"] #normalized pose on robot coordinate
                    ii = random.randint(0, len(pickle_values[ir]["prompt"])-1)
                    inst_obj = pickle_values[ir]["prompt"][ii]
                    inst_obj_x = inst_obj[0]             
                    if isinstance(inst_obj_x, str):
                        flag_data = 1
                    else:     
                        iv = random.randint(0, len(self.image_path)-1)                                        
                except:
                    iv = random.randint(0, len(self.image_path)-1)

            else:
                iv = random.randint(0, len(self.image_path)-1)
            
        voffset = int(224.0*self.v_random*random.random())
        hoffset = int(224.0*self.h_random*random.random())  
        image_obs_list = [] 
        if self.only_front:
            for ih in range(self.context_size + 1):
                image_obs_list.append(self._resize_norm(context_image[ih][:, 0:224, 0:224], self.image_size))    
            goal_image_full_8 = self._resize_norm(goal_image_full_8[:, 0:224, 0:224], self.image_size)   
            PILbox = (hoffset, voffset, 224-hoffset, 224-voffset)
            cropped_image_fullsize_PIL = image_fullsize_PIL.crop(PILbox).resize(self.image_size_clip) 
            cropped_gimage_fullsize_PIL = gimage_fullsize_PIL.crop(PILbox).resize(self.image_size_clip) 
        else:
            for ih in range(self.context_size + 1):     
                image_obs_list.append(self._resize_norm(context_image[ih][:, 0:224, 0:224], self.image_size))    
            goal_image_full_8 = self._resize_norm(goal_image_full_8[:, 0:224, 0:224], self.image_size)       
            PILbox = (hoffset, voffset, 224-hoffset, 224-voffset)
            cropped_image_fullsize_PIL = image_fullsize_PIL.crop(PILbox).resize(self.image_size_clip)            
            cropped_gimage_fullsize_PIL = gimage_fullsize_PIL.crop(PILbox).resize(self.image_size_clip)                               
                                             
        image_obs = torch.cat(image_obs_list[::-1])      
        if random.random() > 0.5:       
            image_obs_r = torch.flip(image_obs, [2])
            goal_image_full_8_r = torch.flip(goal_image_full_8, [2])            
            ob_pose_r = np.array((pose_obj[0], -pose_obj[1]))       
            nomad_traj_norm[:,1] = -nomad_traj_norm[:,1]
            nomad_traj_norm[:,3] = -nomad_traj_norm[:,3]
            
            cropped_image_fullsize_PIL_r = cropped_image_fullsize_PIL.transpose(Image.FLIP_LEFT_RIGHT)            
            cropped_gimage_fullsize_PIL_r = cropped_gimage_fullsize_PIL.transpose(Image.FLIP_LEFT_RIGHT)    
        else:
            image_obs_r = image_obs
            goal_image_full_8_r = goal_image_full_8            
            ob_pose_r = np.array((pose_obj[0], pose_obj[1]))
            cropped_image_fullsize_PIL_r = cropped_image_fullsize_PIL
            cropped_gimage_fullsize_PIL_r = cropped_gimage_fullsize_PIL
            nomad_traj_norm = nomad_traj_norm
                                            
        ob_pose_norm = ob_pose_r/self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
        action_mask = (True)

        thres_dist = 1.5         
        dist_obj = np.sqrt(ob_pose_r[0]**2 + ob_pose_r[1]**2)
        if dist_obj > thres_dist:
            ob_pose_r[0] = ob_pose_r[0]/dist_obj*thres_dist
            ob_pose_r[1] = ob_pose_r[1]/dist_obj*thres_dist   
        
        ob_pose_robot = np.array((ob_pose_r[0], ob_pose_r[1])) #on robot coordinate
        
        dis_obj = np.sqrt(ob_pose_robot[0:1]**2 + ob_pose_robot[1:2]**2)
        metric_waypoint_spacing = 0.25
        obj_pose_norm = np.concatenate((ob_pose_robot[0:1]/metric_waypoint_spacing, ob_pose_robot[1:2]/metric_waypoint_spacing), axis=0)
        goal_pose_cos_sin = np.concatenate((ob_pose_robot[0:1]/metric_waypoint_spacing, ob_pose_robot[1:2]/metric_waypoint_spacing, ob_pose_robot[0:1]/dis_obj, ob_pose_robot[1:2]/dis_obj), axis=0) #Adapting ViNT style action commands (X, Y, cos, sin)            

        # Set the available modality id for each dataset 
        # 0:"satellite only", 1:"pose and satellite", 2:"satellite and image", 3:"all", 4:"pose only", 5:"pose and image", 6:"image only", 7:"language only", 8:"language and pose"
        modality_list = [7, 8]              
        if goal_id == 0:
            if random.random() > 0.5:
                modality_id = random.choice(modality_list)
            else:
                modality_id = 7
        else:
            modality_id = 6   

        ### Adapting OpenVLA stle ###
        actions = nomad_traj_norm
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))
        # Get action chunk string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)
                
        try:
            lang = "move toward " + inst_obj_x
        except:
            print(inst_obj_x) 
            
        if modality_id != 6:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
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
          
        max_token = 60
        if len(input_ids) > max_token:
            try:
                lang = "move toward " + "XXXXX"
            except:
                print(inst_obj_x) 
                  
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
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
        pixel_values = self.image_transform(cropped_image_fullsize_PIL_r)
        pixel_values_g = self.image_transform(cropped_gimage_fullsize_PIL_r)
        
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!     
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
            
        action_chunk_tokens = self.base_tokenizer(action_chunk_string, add_special_tokens=False).input_ids
        action_chunk_len_x = len(action_chunk_tokens)
        dataset_name = "lelan"         
            
        #action select 1.0: raw action, 0.0: MBRA synthetic action            
        if modality_id == 6:        
            action_select_mask = torch.tensor(0.0)
        else:
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
            img_PIL=cropped_image_fullsize_PIL_r,
            gimg_PIL=cropped_gimage_fullsize_PIL_r,
            cur_image = image_obs_r, 
            goal_image_8=goal_image_full_8_r, 
            temp_dist=goal_id,
            lan_prompt=lang
        )  
