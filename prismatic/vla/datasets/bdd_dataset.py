import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple, Type
from tqdm import tqdm
import io
import lmdb
import utm

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import json
import random
import cv2
import matplotlib.pyplot as plt

import math
from typing import Union

from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from vint_train.data.data_utils import (
    img_path_to_data,
    #img_path_to_data_front,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
) 

def img_path_to_data_front(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """
    Load an image from a path and transform it
    Args:
        path (str): path to the image
        image_resize_size (Tuple[int, int]): size to resize the image to
    Returns:
        torch.Tensor: resized image as tensor
    """
    # return transform_images(Image.open(path), transform, image_resize_size, aspect_ratio)
    #return resize_and_aspect_crop_front(Image.open(path), image_resize_size)        
    return TF.to_tensor(Image.open(path))

class LeLaN_Dataset(Dataset):
    def __init__(
        self,
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
        context_type: str = "temporal",
        normalize: bool = True,
        backside: bool = False,
        aug_seq: bool = False,
        only_front: bool = False,                     
    ):
        """
        Main LeLaN dataset class

        Args:
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            image_size (list): Image size 224 x 224
            waypoint_spacing (int): Spacing between waypoints
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            data_split_type (string): train or test 
            data_image_folder (string): path for image
            data_pickle_folder (string): path for pickle file containing object pose, prompts and so on      
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            normalize (bool): Whether to normalize the distances or actions            
            backside (str): whether to use the backsie image or not (basically we can use the backside only for Go Stanford 4 and SACSoN.
            aug_seq (str): whether to use the image before and after.
            only_front (str): whether to contrain the back side image or not 
        """
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
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.normalize = normalize
        self.backside = backside
        self.aug_seq = aug_seq
        self.dataset_name = dataset_name
        self.only_front = only_front

        # load data/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}        
        self._load_split_index()
        self._build_caches_front()
                
        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2   

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches_front()

    def _build_caches_front(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}_{self.data_split_type}.lmdb",
        )

        self._get_augdata()
        
        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    print(cache_filename, len(self.image_path))
                    for num in range(len(self.image_path)):                        
                        if os.path.getsize(self.pickle_path[num]) > 0: 
                            with open(self.image_path[num], "rb") as f:
                                txn.put(self.image_path[num].encode(), f.read())
                        else:
                            print(self.image_path[num])

        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]
        
    def _remove_values_from_list(self, A, B):
        return [item for item in A if item not in B]
            
    def _load_split_index(self):
        if self.dataset_name == "go_stanford4":
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping
            
            lst = os.listdir(self.data_image_folder) # your directory path
            number_files = len(lst)
            
            image_path = []
            pickle_path = []  
            
            ratio = 0.9
            thres = int(number_files*ratio)
            
            if self.data_split_type == "train":
                print("go_stanford4 train flame num", thres)
            else:
                print("go_stanford4 test flame num", number_files-thres)            
             
            #TODO -5 is come from "self.data_image_folder" includes 5 files, which is not pickle file.
            for num in range(int(number_files - 5)-3):
                if self.data_split_type == "train" and num < thres:
                    image_path.append(self.data_image_folder + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + str(num).zfill(8) + '.pkl')                               
                elif self.data_split_type == "test" and num >= thres:
                    image_path.append(self.data_image_folder + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + str(num).zfill(8) + '.pkl')                                     
      
        if self.dataset_name == "sacson":
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping 
                        
            image_path = []
            pickle_path = []   
                                     
            folder_lst = next(os.walk(self.data_pickle_folder))[1]

            if self.data_split_type == "train":
                folder_lst_dataset = folder_lst[0:len(folder_lst)-1]
                print("SACSoN dataset train seq. number", len(folder_lst_dataset))
            else:
                folder_lst_dataset = folder_lst[len(folder_lst)-1:len(folder_lst)]
                print("SACSoN dataset test seq. number", len(folder_lst_dataset))
                                
            for folder in folder_lst_dataset:
                print(self.data_split_type, folder)
                subfolder_lst = os.listdir(self.data_pickle_folder + folder + "/")                                    
                for subfolder in subfolder_lst:
                    file_lst = os.listdir(self.data_image_folder + folder + "/" + subfolder + "/image/")
                    number_files = len(file_lst)
                    for num in range(int(number_files)-3):
                        image_path.append(self.data_image_folder + folder + "/" + subfolder + "/image/" + str(num).zfill(8) + '.jpg')
                        pickle_path.append(self.data_pickle_folder + folder + "/" + subfolder + "/pickle/" + str(num).zfill(8) + '.pkl')                        

        if self.dataset_name == "go_stanford2":
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping 
                        
            image_path = []
            pickle_path = [] 
                                     
            folder_lst = next(os.walk(self.data_pickle_folder))[1]
            num_test = int(0.1*len(folder_lst))
            
            if self.data_split_type == "train":
                folder_lst_dataset = folder_lst[0:len(folder_lst)-num_test]
                print("go_stanford2 train seq. number", len(folder_lst_dataset))
            else:
                folder_lst_dataset = folder_lst[len(folder_lst)-num_test:len(folder_lst)]
                print("go_stanford2 test seq. number", len(folder_lst_dataset))
            
            for folder in folder_lst_dataset:
                file_lst = os.listdir(self.data_image_folder + folder + "/image/")
                number_files = len(file_lst)
                for num in range(int(number_files-3)):
                    image_path.append(self.data_image_folder + folder + "/image/" + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + folder + "/pickle/" + str(num).zfill(8) + '.pkl')                               

        if self.dataset_name == "humanw":
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping 
                    
            image_path = []
            pickle_path = []                                     
            folder_lst = next(os.walk(self.data_pickle_folder))[1]            
            test_folder = ["R0010096", "R0010098","R0010121", "R0010118","R0010133", "R0010145", "R0010156", "R0010166", "R0010175","R0010180", "R0010188", "R0010197"]
            
            if self.data_split_type == "train":
                folder_lst_dataset = self._remove_values_from_list(folder_lst, test_folder)
                print("Human-walking dataset train seq. number", len(folder_lst_dataset))
            else:
                folder_lst_dataset = test_folder
                print("Human-walking dataset test seq. number", len(folder_lst_dataset))
            
            for folder in folder_lst_dataset:
                file_lst = os.listdir(self.data_image_folder + folder + "/image/")
                number_files = len(file_lst)
                for num in range(int(number_files)):
                    image_path.append(self.data_image_folder + folder + "/image/" + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + folder + "/pickle/" + str(num).zfill(8) + '.pkl')                         

        if self.dataset_name == "youtube":
            self.v_random = 0.05 #for random cropping
            self.h_random = 0.05 #for random cropping 
                    
            image_path = []
            pickle_path = []                                       
            folder_lst = next(os.walk(self.data_pickle_folder))[1]
            test_folder = ["home_10", "home_12", "austra_1", "spain_1", "singa_1", "spain_3", "spain_5", "rosia_2", "home_33", "poland_1", "uk_5"]
              
            if self.data_split_type == "train":
                folder_lst_dataset = self._remove_values_from_list(folder_lst, test_folder)
                print("YouTube dataset train seq. number", len(folder_lst_dataset))                
            else:
                folder_lst_dataset = test_folder
                print("YouTube dataset test seq. number", len(folder_lst_dataset))   
            
            for folder in folder_lst_dataset:
                file_lst = os.listdir(self.data_image_folder + folder + "/image/")
                number_files = len(file_lst)
                for num in range(int(number_files)):
                    image_path.append(self.data_image_folder + folder + "/image/" + str(num).zfill(8) + '.jpg')
                    pickle_path.append(self.data_pickle_folder + folder + "/pickle/" + str(num).zfill(8) + '.pkl')                        
          
        self.image_path = image_path
        self.pickle_path = pickle_path

    def _load_image_front(self, path):
        image_path = path
        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())               
                image_bytes = bytes(image_buffer)             
            image_bytes = io.BytesIO(image_bytes)         
            return img_path_to_data_front(image_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load image {image_path}")

    def _resize_norm(self, image, size):
        return TF.resize(image, size)
        #return TF.resize(2.0*(image - 0.5), size)
        
    def _get_augdata(self, ):
        aug_data_list = []
        for num in range(len(self.pickle_path)):
            if os.path.getsize(self.pickle_path[num]) > 0:            
                with open(self.pickle_path[num], "rb") as f:
                    aug_data = pickle.load(f)
            else:
                print(self.pickle_path[num])
            aug_data_list.append(aug_data)
            
        self.aug_data_list = aug_data_list

    def __len__(self) -> int:
        return len(self.image_path)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image 
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        flag_data = 0
        flag_data_inner = 0
        iv = i
        
        ib = random.random()
        flag_back = 0
        
        if self.backside and ib > 0.5:
            flag_back = 1
        
        #remove data without object in front of the robot
        while flag_data == 0:
            image_fullsize = self._load_image_front(self.image_path[iv])
            flag_data_inner = 0
            
            context_image = [image_fullsize]        
            for ih in range(self.context_size):
                if iv-ih > 0:                   
                    context_image.append(self._load_image_front(self.image_path[iv-ih]))             
                else:
                    context_image.append(self._load_image_front(self.image_path[0]))
            
            for ih in range(self.context_size + 1):
                if context_image[ih] is None:
                    flag_data_inner = 1
                    iv = random.randint(0, len(self.image_path)-1)
      
            pickle_values = self.aug_data_list[iv]                
            if len(pickle_values) != 0:
                list_rand = [random.randint(0, len(pickle_values)-1) for i in range(len(pickle_values))]            
                il = 0
                c_pose_check = 0
                while flag_data_inner == 0:
                    ir = list_rand[il]
                    if flag_back == 0: #flag_back = 0 --> front-side, flag_back = 1 --> back-side
                        thres_data = pickle_values[ir]["bbox"][3] <= 224 and pickle_values[ir]["obj_detect"]
                    else:
                        thres_data = pickle_values[ir]["bbox"][2] >= 224 and pickle_values[ir]["obj_detect"]
       
                    if thres_data:                                           
                        if 0 <= pickle_values[ir]["bbox"][0] and pickle_values[ir]["bbox"][0] < 224-1:
                            bbox_top = int(pickle_values[ir]["bbox"][0])
                        elif pickle_values[ir]["bbox"][0] < 0:
                            bbox_top = 0                        
                        else:
                            bbox_top = 223
                        if 0 <= pickle_values[ir]["bbox"][1] and pickle_values[ir]["bbox"][1] < 224-1:
                            bbox_bottom = int(pickle_values[ir]["bbox"][1])
                        elif pickle_values[ir]["bbox"][1] < 0:
                            bbox_bottom = 0                        
                        else:
                            bbox_bottom = 223
                        if 0 <= pickle_values[ir]["bbox"][2] and pickle_values[ir]["bbox"][2] < 224-1:
                            bbox_left = int(pickle_values[ir]["bbox"][2])
                        elif pickle_values[ir]["bbox"][2] < 0:
                            bbox_left = 0                        
                        else:
                            bbox_left = 223
                        if 0 <= pickle_values[ir]["bbox"][3] and pickle_values[ir]["bbox"][3] < 224-1:
                            bbox_right = int(pickle_values[ir]["bbox"][3])
                        elif pickle_values[ir]["bbox"][3] < 0:
                            bbox_right = 0                        
                        else:
                            bbox_right = 223
                                                                                
                        image_crop = image_fullsize[:, bbox_top:bbox_bottom, bbox_left:bbox_right]                        
                        if flag_back == 0:
                            pose_obj = pickle_values[ir]["pose_median"]
                        else:
                            pose_obj = [-pickle_values[ir]["pose_median"][0], pickle_values[ir]["pose_median"][1], -pickle_values[ir]["pose_median"][2]]
                        
                        flag_text = 0
                        if "prompt" in pickle_values[ir].keys():
                            ii = random.randint(0, len(pickle_values[ir]["prompt"])-1)
                            inst_obj = pickle_values[ir]["prompt"][ii]

                            if isinstance(inst_obj, list):
                                flag_text = 0
                            else:
                                flag_text = 1                                                                                                           
                        inst_obj_x = inst_obj

                        if pickle_values[ir]["pose_median"][0]**2 + pickle_values[ir]["pose_median"][2]**2 > 10.0**2 or flag_text == 0: 
                            c_pose_check += 1
                            if c_pose_check == 5:                  
                                flag_data_inner = 1
                                iv = random.randint(0, len(self.image_path)-1)
                            else:
                                flag_data_inner = 0                                
                        else:
                            flag_data_inner = 1 
                            flag_data = 1                                                                             
                    else:
                        il += 1
                        if il+1 > len(list_rand):
                            flag_data_inner = 1
                            iv = random.randint(0, len(self.image_path)-1)
            else:
                iv = random.randint(0, len(self.image_path)-1)
            
        voffset = int(224.0*self.v_random*random.random())
        hoffset = int(224.0*self.h_random*random.random())

        image_obs_list = [] 
        if self.only_front:
            for ih in range(self.context_size + 1):
                image_obs_list.append(self._resize_norm(context_image[ih][:, voffset:224-voffset, hoffset:224-hoffset], self.image_size))                
        else:
            if flag_back == 0:
                for ih in range(self.context_size + 1):     
                    image_obs_list.append(self._resize_norm(context_image[ih][:, voffset:224-voffset, hoffset:224-hoffset], self.image_size))              
            else:
                for ih in range(self.context_size + 1):  
                    image_obs_list.append(self._resize_norm(context_image[ih][:, voffset:224-voffset, 224+hoffset:2*224-hoffset], self.image_size))                   
        image_obs = torch.cat(image_obs_list[::-1])      
        image_crop = self._resize_norm(image_crop, self.image_size)        
       
        if random.random() > 0.5:
            image_obs_r = torch.flip(image_obs, [2])
            image_crop_r = torch.flip(image_crop, [2])
            ob_pose_r = np.array((-pose_obj[0], pose_obj[2]))
        else:
            image_obs_r = image_obs
            image_crop_r = image_crop
            ob_pose_r = np.array((pose_obj[0], pose_obj[2]))        
        ob_pose_norm = ob_pose_r/self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
        
        return (
            torch.as_tensor(image_obs_r, dtype=torch.float32),
            torch.as_tensor(image_crop_r, dtype=torch.float32),
            torch.as_tensor(ob_pose_r, dtype=torch.float32),
            inst_obj_x,
            torch.as_tensor(ob_pose_norm, dtype=torch.float32),
        )         
        
class BDD_Dataset_multi(Dataset):
    def __init__(
        self,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        data_split_type: str,
        data_folder: str,        
        #data_image_folder: str,
        #data_pickle_folder: str,       
        #lan_solo: bool = False,          
        context_type: str = "temporal",
        normalize: bool = True,
        model_type: str = "MBRA",
        #backside: bool = False,
        aug_seq: bool = False,
        #only_front: bool = False,                     
    ):
        """
        Main LeLaN dataset class

        Args:
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            image_size (list): Image size 224 x 224
            waypoint_spacing (int): Spacing between waypoints
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            data_split_type (string): train or test 
            data_image_folder (string): path for image
            data_pickle_folder (string): path for pickle file containing object pose, prompts and so on      
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            normalize (bool): Whether to normalize the distances or actions            
            backside (str): whether to use the backsie image or not (basically we can use the backside only for Go Stanford 4 and SACSoN.
            aug_seq (str): whether to use the image before and after.
            only_front (str): whether to contrain the back side image or not 
        """
        self.data_split_folder = data_split_folder
        self.data_split_type = data_split_type
        #self.data_image_folder = data_image_folder
        #self.data_pickle_folder = data_pickle_folder    
        self.data_folder = data_folder                
        self.image_size = image_size
        
        if model_type == "MBRA":
            self.image_size_clip = (128, 416)
        else:
            self.image_size_clip = (224, 224)
        self.waypoint_spacing = waypoint_spacing
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.normalize = normalize
        #self.backside = backside
        self.aug_seq = aug_seq
        self.dataset_name = dataset_name
        #self.only_front = only_front

        # load data/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}        
        self._load_split_index()
        self._build_caches_front()
        #self.lan_solo = lan_solo
        
        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2   

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches_front()

    def _build_caches_front(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}_{self.data_split_type}.lmdb",
        )

        #self._get_augdata()
        
        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    print(cache_filename, len(self.image_path))
                    for num in range(len(self.image_path)): 
                        for ih in range(self.context_size):
                            with open(self.image_path[num][ih], "rb") as f:
                                txn.put(self.image_path[num][ih].encode(), f.read())                    
                        """                       
                        if os.path.getsize(self.pickle_path[num]) > 0: 
                            with open(self.image_path[num], "rb") as f:
                                txn.put(self.image_path[num].encode(), f.read())
                        else:
                            print(self.image_path[num])
                        """
        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]
        
    def _remove_values_from_list(self, A, B):
        return [item for item in A if item not in B]
            
    def _load_split_index(self):
        if self.dataset_name == "bdd":
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping 
                        
            image_path = []
            gps_list = []   
                                     
            folder_lst = next(os.walk(self.data_folder))[1]

            if self.data_split_type == "train":
                folder_lst_dataset = folder_lst[0:len(folder_lst)-int(len(folder_lst)*0.1)]
                print("BDD dataset train seq. number", len(folder_lst_dataset))
            else:
                folder_lst_dataset = folder_lst[len(folder_lst)-int(len(folder_lst)*0.1):len(folder_lst)]
                print("BDD dataset test seq. number", len(folder_lst_dataset))
                                
            for folder in folder_lst_dataset:
                print(self.data_split_type, folder)
                with open(self.data_folder + "/" + folder + "/gps.json", 'r', encoding='utf-8') as f:
                    gps_data = json.load(f)   
                gps_list += gps_data                          
                #for data_img in gps_data:
                #    number_files = len(gps_data)
                number_files = len(gps_data)
                for num in range(number_files):
                    #print(gps_data[num]["img_id"])
                    image_path_hist = []                  
                    for ih in range(self.context_size):
                        if gps_data[num]["img_id"] - ih > 0:
                            image_path_hist.append(self.data_folder + "/" + folder + "/img/" + str(gps_data[num]["img_id"] - ih) + '.jpg')       
                        else:
                            image_path_hist.append(self.data_folder + "/" + folder + "/img/" + str(0) + '.jpg')   
                            
                                                                                           
                    image_path.append(image_path_hist)
                        
        self.image_path = image_path
        self.gps_list = gps_list
        sum_num = 0
        for ic in range(len(self.image_path)):
            sum_num += len(self.image_path[ic])
            #print(self.image_path[ic])
            if len(self.image_path[ic]) != 5:
                print(self.image_path[ic])
        
        print("check BDD", len(self.image_path), len(self.gps_list), len(self.image_path[0]), sum_num)

    def _load_image_front(self, path):
        image_path = path
        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())               
                image_bytes = bytes(image_buffer)             
            image_bytes = io.BytesIO(image_bytes)         
            return img_path_to_data_front(image_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load image {image_path}")

    def _resize_norm(self, image, size):
        return TF.resize(image, size)
        #return TF.resize(2.0*(image - 0.5), size)
    """    
    def _get_augdata(self, ):
        aug_data_list = []
        for num in range(len(self.pickle_path)):
            if os.path.getsize(self.pickle_path[num]) > 0:            
                with open(self.pickle_path[num], "rb") as f:
                    aug_data = pickle.load(f)
            else:
                print(self.pickle_path[num])
            aug_data_list.append(aug_data)
            
        self.aug_data_list = aug_data_list
    """
    def _transform_position(self, lat, lon, heading, X, Y, theta):
        """
        Compute new latitude, longitude, and heading after moving by (X, Y, theta)
        in the local coordinate system where:
        - X is forward (aligned with heading)
        - Y is left (perpendicular to heading)
        - theta is counterclockwise (CCW)
        """
        # Convert lat/lon to UTM
        easting, northing, zone_number, zone_letter = self._latlon_to_utm(lat, lon)

        # Convert heading from degrees to radians
        # heading_rad = np.radians(heading)
        heading_rad = heading
        #new_heading = (heading - theta) 
        new_heading = (heading + theta) 
    
        # Corrected transformation: X moves forward, Y moves left
        #delta_easting = np.sqrt(X**2 + Y**2) * np.sin(new_heading) 
        #delta_northing = np.sqrt(X**2 + Y**2) * np.cos(new_heading) 
        delta_northing = X*np.cos(heading_rad) - Y*np.sin(heading_rad)
        delta_easting = -X*np.sin(heading_rad) - Y*np.cos(heading_rad)
    
        # New position in UTM coordinates
        new_easting = easting + delta_easting
        new_northing = northing + delta_northing

        # Convert back to latitude and longitude
        new_lat, new_lon = self._utm_to_latlon(new_easting, new_northing, zone_number, zone_letter)

        # Update heading (subtract for CCW rotation)
        #new_heading = (heading - theta)  
        new_heading = (heading + theta) 
        
        return new_lat, new_lon, new_heading    

    def _calculate_relative_position(self, x_a, y_a, x_b, y_b):
        delta_x = x_b - x_a
        delta_y = y_b - y_a
        return delta_x, delta_y

    def _rotate_to_local_frame(self, delta_x, delta_y, heading_a_rad):  
        # Apply the rotation matrix for the local frame
        relative_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        relative_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return relative_x, relative_y
            
    def __len__(self) -> int:
        return len(self.image_path)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image 
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        flag_data = 0
        iv = i
        
        it = random.random()
        #flag_back = 0
        #
        #if self.backside and ib > 0.5:
        #    flag_back = 1
        
        #remove data without object in front of the robot
        while flag_data == 0:
            image_fullsize = self._load_image_front(self.image_path[iv][0])
            #flag_data_inner = 0
            
            context_image = [image_fullsize]        
            for ih in range(self.context_size):
                context_image.append(self._load_image_front(self.image_path[iv][ih]))             
            """
            for ih in range(self.context_size + 1):
                if context_image[ih] is None:
                    flag_data_inner = 1
                    iv = random.randint(0, len(self.image_path)-1)
            """
            #goal_id = random.randint(0,5)
            goal_id = 1
            cur_pos = self.gps_list[iv]                
            try:
                #print(f"Accessing image_path at index {iv + goal_id}")
                #print("goal path", self.image_path[iv + goal_id][0])
                #print("cur path", self.image_path[iv][0])
                goal_image_full = self._load_image_front(self.image_path[iv + goal_id][0])
                goal_image_full_8 = self._load_image_front(self.image_path[iv + 1][0])     
                
                curr_path = self.image_path[iv][0]   
                goal_path = self.image_path[iv + 1][0]           
                goal_pos = self.gps_list[iv + goal_id]
            except:
                #print(f"[EXCEPT] Caught exception: {e}")
                goal_id = 0
                goal_image_full = self._load_image_front(self.image_path[iv + goal_id][0])   
                goal_image_full_8 = goal_image_full
                goal_pos = self.gps_list[iv + goal_id]     
                curr_path = self.image_path[iv][0]   
                goal_path = self.image_path[iv + goal_id][0]   
                            
            cur_compass = -cur_pos["course"]/180.0*3.141592
            goal_compass = -goal_pos["course"]/180.0*3.141592
                
            cur_utm = utm.from_latlon(cur_pos["latitude"], cur_pos["longitude"])
            goal_utm = utm.from_latlon(goal_pos["latitude"], goal_pos["longitude"])
            delta_x, delta_y = self._calculate_relative_position(cur_utm[0], cur_utm[1], goal_utm[0], goal_utm[1])
            relative_x, relative_y = self._rotate_to_local_frame(delta_x, delta_y, cur_compass)
            radius = np.sqrt(relative_x**2 + relative_y**2)
            ## Clipping the goal pose with thres_dist
            #radius = np.sqrt(relative_x**2 + relative_y**2)
            #if radius > thres_dist:
            #    relative_x = relative_x/radius*thres_dist
            #    relative_y = relative_y/radius*thres_dist                 
            
            
            if it > 0.5 and abs(goal_compass - cur_compass) < 3.141592*20.0/180.0:
                iv = random.randint(0, len(self.image_path)-1)
            else:
                flag_data = 1
            
            curr_path_list = curr_path.split("/")
            goal_path_list = goal_path.split("/")            
            limit_radius = 3.0    
            #print(curr_path_list[-3], goal_path_list[-3])
            #if radius > 15.0 and curr_path_list[-3] != goal_path_list[-3]:
            #    print(radius, curr_path_list[-3], goal_path_list[-3])
            if radius > 15.0 or curr_path_list[-3] != goal_path_list[-3]:
                iv = random.randint(0, len(self.image_path)-1)
                flag_data = 0
            else:
                #print(relative_x, relative_y)    
                if radius > limit_radius and abs(goal_compass - cur_compass) < 3.141592*30.0/180.0:
                    #print("before", relative_x, relative_y)                
                    relative_x = relative_x/radius*limit_radius
                    relative_y = relative_y/radius*limit_radius
                    #print("after", relative_x, relative_y)
            #flag_data = 1
            
        voffset = int(224.0*self.v_random*random.random())
        hoffset = int(224.0*self.h_random*random.random())

        image_obs_list = [] 
        image_crop_list = [] 
        metric_waypoint_spacing = 0.25 #normalization 
        for ih in range(self.context_size + 1):
            image_crop_list.append(self._resize_norm(context_image[ih][:, voffset:224-voffset, hoffset:224-hoffset], self.image_size))  
            image_obs_list.append(self._resize_norm(context_image[ih], self.image_size))     
        goal_image_full = self._resize_norm(goal_image_full[:, voffset:224-voffset, hoffset:224-hoffset], self.image_size)        
        goal_image_full_8 = self._resize_norm(goal_image_full_8, self.image_size) 
        cur_image_large = self._resize_norm(context_image[0][:, voffset:224-voffset, hoffset:224-hoffset], self.image_size_clip)                 
                              
        image_obs = torch.cat(image_obs_list[::-1])      
        image_crop = torch.cat(image_crop_list[::-1])           
        #image_crop = self._resize_norm(image_crop, self.image_size)        
       
        if random.random() > 0.5:
            image_obs_r = torch.flip(image_obs, [2])
            image_crop_r = torch.flip(image_crop, [2])
            cur_image_large_r = torch.flip(cur_image_large, [2])
            goal_image_full_r = torch.flip(goal_image_full, [2])
            goal_image_full_8_r = torch.flip(goal_image_full_8, [2])            
            goal_pose_r = np.array([relative_y/metric_waypoint_spacing, relative_x/metric_waypoint_spacing, np.cos(-goal_compass+cur_compass), np.sin(-goal_compass+cur_compass)])      
        else:
            image_obs_r = image_obs
            image_crop_r = image_crop
            cur_image_large_r = cur_image_large
            goal_image_full_r = goal_image_full
            goal_image_full_8_r = goal_image_full_8            
            goal_pose_r = np.array([relative_y/metric_waypoint_spacing, -relative_x/metric_waypoint_spacing, np.cos(goal_compass-cur_compass), np.sin(goal_compass-cur_compass)])         
        #ob_pose_norm = ob_pose_r/self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
        action_mask = (True)
        
        goad_id_virtual = -1
        for ig in range(20):
            if 0.3333*ig < radius and radius < 0.3333*(ig + 1):
                goad_id_virtual = ig
        if goad_id_virtual == -1:
            goad_id_virtual = 20
        if goal_id > 1:
            goad_id_virtual = 100
                                    
        return (
            torch.as_tensor(image_obs_r, dtype=torch.float32),
            torch.as_tensor(image_crop_r, dtype=torch.float32),
            torch.as_tensor(cur_image_large_r, dtype=torch.float32),
            torch.as_tensor(goal_pose_r, dtype=torch.float32),
            torch.as_tensor(goal_image_full_r, dtype=torch.float32),   
            torch.as_tensor(goal_image_full_8_r, dtype=torch.float32),                      
            torch.as_tensor(goad_id_virtual, dtype=torch.float32),  
            torch.as_tensor(action_mask, dtype=torch.float32), 
            curr_path,           
            goal_path, 
        )                   

class BDD_Dataset_multi_MMN(Dataset):
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
        #data_image_folder: str,
        #data_pickle_folder: str,       
        #lan_solo: bool = False,       
        predict_stop_token: bool = True,    
        context_type: str = "temporal",
        normalize: bool = True,
        model_type: str = "MBRA",
        #backside: bool = False,
        aug_seq: bool = False,
        #only_front: bool = False,                     
    ):
        """
        Main LeLaN dataset class

        Args:
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            image_size (list): Image size 224 x 224
            waypoint_spacing (int): Spacing between waypoints
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            data_split_type (string): train or test 
            data_image_folder (string): path for image
            data_pickle_folder (string): path for pickle file containing object pose, prompts and so on      
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            normalize (bool): Whether to normalize the distances or actions            
            backside (str): whether to use the backsie image or not (basically we can use the backside only for Go Stanford 4 and SACSoN.
            aug_seq (str): whether to use the image before and after.
            only_front (str): whether to contrain the back side image or not 
        """
        self.data_split_folder = data_split_folder
        self.data_split_type = data_split_type
        #self.data_image_folder = data_image_folder
        #self.data_pickle_folder = data_pickle_folder    
        self.data_folder = data_folder        
        self.pickle_folder = pickle_folder                   
        self.image_size = image_size
        
        if model_type == "MBRA":
            self.image_size_clip = (128, 416)
        else:
            self.image_size_clip = (224, 224)
        self.waypoint_spacing = waypoint_spacing
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.normalize = normalize
        #self.backside = backside
        self.aug_seq = aug_seq
        self.dataset_name = dataset_name
        #self.only_front = only_front

        # load data/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}        
        self._load_split_index()
        self._build_caches_front()
        #self.lan_solo = lan_solo
        
        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2   

        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.prompt_builder = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.image_transform = image_transform            

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches_front()

    def _build_caches_front(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}_{self.data_split_type}.lmdb",
        )

        self._get_mbra_traj()
        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    print(cache_filename, len(self.image_path))
                    for num in range(len(self.image_path)): 
                        for ih in range(self.context_size):
                            with open(self.image_path[num][ih], "rb") as f:
                                txn.put(self.image_path[num][ih].encode(), f.read())                    
                        """                       
                        if os.path.getsize(self.pickle_path[num]) > 0: 
                            with open(self.image_path[num], "rb") as f:
                                txn.put(self.image_path[num].encode(), f.read())
                        else:
                            print(self.image_path[num])
                        """
        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True, max_readers=2048)

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]
        
    def _remove_values_from_list(self, A, B):
        return [item for item in A if item not in B]
            
    def _load_split_index(self):
        if self.dataset_name == "bdd":
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping 
                        
            image_path = []
            gps_list = []   
            pickle_path = []
                                                 
            folder_lst = next(os.walk(self.data_folder))[1]

            if self.data_split_type == "train":
                folder_lst_dataset = folder_lst[0:len(folder_lst)-int(len(folder_lst)*0.1)]
                print("BDD dataset train seq. number", len(folder_lst_dataset))
            else:
                folder_lst_dataset = folder_lst[len(folder_lst)-int(len(folder_lst)*0.1):len(folder_lst)]
                print("BDD dataset test seq. number", len(folder_lst_dataset))
                                
            for folder in tqdm(folder_lst_dataset):
                #print(self.data_split_type, folder)
                with open(self.data_folder + "/" + folder + "/gps.json", 'r', encoding='utf-8') as f:
                    gps_data = json.load(f)   
                gps_list += gps_data                          
                #for data_img in gps_data:
                #    number_files = len(gps_data)
                number_files = len(gps_data)
                for num in range(number_files):
                    #print(gps_data[num]["img_id"])
                    image_path_hist = []                  
                    for ih in range(self.context_size):
                        if gps_data[num]["img_id"] - ih > 0:
                            image_path_hist.append(self.data_folder + "/" + folder + "/img/" + str(gps_data[num]["img_id"] - ih) + '.jpg')       
                        else:
                            image_path_hist.append(self.data_folder + "/" + folder + "/img/" + str(0) + '.jpg')                                                                  
                    image_path.append(image_path_hist)
                    pickle_path.append(self.pickle_folder + "/" + folder + "/pickle/" + str(gps_data[num]["img_id"]) + '.pkl')
                        
        self.image_path = image_path
        self.pickle_path = pickle_path
        self.gps_list = gps_list
        sum_num = 0
        for ic in range(len(self.image_path)):
            sum_num += len(self.image_path[ic])
            #print(self.image_path[ic])
            if len(self.image_path[ic]) != 5:
                print(self.image_path[ic])
        
        print("check BDD", len(self.image_path), len(self.gps_list), len(self.image_path[0]), sum_num)

    def _get_mbra_traj(self, ):
        mbra_traj_list = []
        for num in tqdm(range(len(self.pickle_path)), desc="BDD pickle loading"):
            if os.path.getsize(self.pickle_path[num]) > 0:            
                with open(self.pickle_path[num], "rb") as f:
                    aug_data = pickle.load(f)
            else:
                print(self.pickle_path[num])
            mbra_traj_list.append(aug_data)
            
        self.mbra_traj_list = mbra_traj_list

    def _load_image_front(self, path):
        image_path = path
        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())               
                image_bytes = bytes(image_buffer)             
            image_bytes = io.BytesIO(image_bytes)         
            return img_path_to_data_front(image_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load image {image_path}")

    def _resize_norm(self, image, size):
        return TF.resize(image, size)
        #return TF.resize(2.0*(image - 0.5), size)
    """    
    def _get_augdata(self, ):
        aug_data_list = []
        for num in range(len(self.pickle_path)):
            if os.path.getsize(self.pickle_path[num]) > 0:            
                with open(self.pickle_path[num], "rb") as f:
                    aug_data = pickle.load(f)
            else:
                print(self.pickle_path[num])
            aug_data_list.append(aug_data)
            
        self.aug_data_list = aug_data_list
    """
    def _transform_position(self, lat, lon, heading, X, Y, theta):
        """
        Compute new latitude, longitude, and heading after moving by (X, Y, theta)
        in the local coordinate system where:
        - X is forward (aligned with heading)
        - Y is left (perpendicular to heading)
        - theta is counterclockwise (CCW)
        """
        # Convert lat/lon to UTM
        easting, northing, zone_number, zone_letter = self._latlon_to_utm(lat, lon)

        # Convert heading from degrees to radians
        # heading_rad = np.radians(heading)
        heading_rad = heading
        #new_heading = (heading - theta) 
        new_heading = (heading + theta) 
    
        # Corrected transformation: X moves forward, Y moves left
        #delta_easting = np.sqrt(X**2 + Y**2) * np.sin(new_heading) 
        #delta_northing = np.sqrt(X**2 + Y**2) * np.cos(new_heading) 
        delta_northing = X*np.cos(heading_rad) - Y*np.sin(heading_rad)
        delta_easting = -X*np.sin(heading_rad) - Y*np.cos(heading_rad)
    
        # New position in UTM coordinates
        new_easting = easting + delta_easting
        new_northing = northing + delta_northing

        # Convert back to latitude and longitude
        new_lat, new_lon = self._utm_to_latlon(new_easting, new_northing, zone_number, zone_letter)

        # Update heading (subtract for CCW rotation)
        #new_heading = (heading - theta)  
        new_heading = (heading + theta) 
        
        return new_lat, new_lon, new_heading    

    def _calculate_relative_position(self, x_a, y_a, x_b, y_b):
        delta_x = x_b - x_a
        delta_y = y_b - y_a
        return delta_x, delta_y

    def _rotate_to_local_frame(self, delta_x, delta_y, heading_a_rad):  
        # Apply the rotation matrix for the local frame
        relative_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        relative_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return relative_x, relative_y
            
    def __len__(self) -> int:
        return len(self.image_path)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image 
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        flag_data = 0
        iv = i
        
        it = random.random()
        #flag_back = 0
        #
        #if self.backside and ib > 0.5:
        #    flag_back = 1
        
        #remove data without object in front of the robot
        while flag_data == 0:
            image_fullsize = self._load_image_front(self.image_path[iv][0])
            PIL_current = to_pil_image(image_fullsize).resize((224, 224))
            #flag_data_inner = 0
            
            context_image = [image_fullsize]        
            for ih in range(self.context_size):
                context_image.append(self._load_image_front(self.image_path[iv][ih]))             
            """
            for ih in range(self.context_size + 1):
                if context_image[ih] is None:
                    flag_data_inner = 1
                    iv = random.randint(0, len(self.image_path)-1)
            """
            #goal_id = random.randint(0,5)
            goal_id = 1
            cur_pos = self.gps_list[iv]    
            mbra_traj = self.mbra_traj_list[iv]                        
            try:
                #print(f"Accessing image_path at index {iv + goal_id}")
                #print("goal path", self.image_path[iv + goal_id][0])
                #print("cur path", self.image_path[iv][0])
                goal_image_full = self._load_image_front(self.image_path[iv + goal_id][0])
                goal_image_full_8 = self._load_image_front(self.image_path[iv + 1][0])     
                PIL_goal = to_pil_image(goal_image_full).resize((224, 224))
                
                curr_path = self.image_path[iv][0]   
                goal_path = self.image_path[iv + 1][0]           
                goal_pos = self.gps_list[iv + goal_id]
            except:
                #print(f"[EXCEPT] Caught exception: {e}")
                goal_id = 0
                goal_image_full = self._load_image_front(self.image_path[iv + goal_id][0])   
                goal_image_full_8 = goal_image_full
                PIL_goal = to_pil_image(goal_image_full).resize((224, 224))
                goal_pos = self.gps_list[iv + goal_id]     
                curr_path = self.image_path[iv][0]   
                goal_path = self.image_path[iv + goal_id][0]   
                            
            #cur_compass = -cur_pos["course"]/180.0*3.141592
            #goal_compass = -goal_pos["course"]/180.0*3.141592
            try:                
                cur_compass = -cur_pos["course"]/180.0*3.141592
                goal_compass = -goal_pos["course"]/180.0*3.141592
            except:
                print("BDD dataset no key course")
                cur_compass = 0.0/180.0*3.141592
                goal_compass = 0.0/180.0*3.141592
                                
            cur_utm = utm.from_latlon(cur_pos["latitude"], cur_pos["longitude"])
            goal_utm = utm.from_latlon(goal_pos["latitude"], goal_pos["longitude"])
            delta_x, delta_y = self._calculate_relative_position(cur_utm[0], cur_utm[1], goal_utm[0], goal_utm[1])
            relative_x, relative_y = self._rotate_to_local_frame(delta_x, delta_y, cur_compass)
            radius = np.sqrt(relative_x**2 + relative_y**2)
            ## Clipping the goal pose with thres_dist
            #radius = np.sqrt(relative_x**2 + relative_y**2)
            #if radius > thres_dist:
            #    relative_x = relative_x/radius*thres_dist
            #    relative_y = relative_y/radius*thres_dist                 
            
            if it > 0.5 and abs(goal_compass - cur_compass) < 3.141592*20.0/180.0:
                iv = random.randint(0, len(self.image_path)-1)
            else:
                flag_data = 1
            
            curr_path_list = curr_path.split("/")
            goal_path_list = goal_path.split("/")            
            limit_radius = 3.0    
            #print(curr_path_list[-3], goal_path_list[-3])
            #if radius > 15.0 and curr_path_list[-3] != goal_path_list[-3]:
            #    print(radius, curr_path_list[-3], goal_path_list[-3])
            if radius > 15.0 or curr_path_list[-3] != goal_path_list[-3]:
                iv = random.randint(0, len(self.image_path)-1)
                flag_data = 0
            else:
                #print(relative_x, relative_y)    
                if radius > limit_radius and abs(goal_compass - cur_compass) < 3.141592*30.0/180.0:
                    #print("before", relative_x, relative_y)                
                    relative_x = relative_x/radius*limit_radius
                    relative_y = relative_y/radius*limit_radius
                    #print("after", relative_x, relative_y)
            #flag_data = 1
            
        voffset = int(224.0*self.v_random*random.random())
        hoffset = int(224.0*self.h_random*random.random())

        image_obs_list = [] 
        image_crop_list = [] 
        metric_waypoint_spacing = 0.25 #normalization 
        for ih in range(self.context_size + 1):
            image_crop_list.append(self._resize_norm(context_image[ih][:, voffset:224-voffset, hoffset:224-hoffset], self.image_size))  
            image_obs_list.append(self._resize_norm(context_image[ih], self.image_size))     
        goal_image_full = self._resize_norm(goal_image_full[:, voffset:224-voffset, hoffset:224-hoffset], self.image_size)        
        goal_image_full_8 = self._resize_norm(goal_image_full_8, self.image_size) 
        cur_image_large = self._resize_norm(context_image[0][:, voffset:224-voffset, hoffset:224-hoffset], self.image_size_clip)                 

        PILbox = (hoffset, voffset, 224-hoffset, 224-voffset)
        PIL_current_crop = PIL_current.crop(PILbox).resize(self.image_size_clip) 
        PIL_goal_crop = PIL_goal.crop(PILbox).resize(self.image_size_clip) 
                                          
        image_obs = torch.cat(image_obs_list[::-1])      
        image_crop = torch.cat(image_crop_list[::-1])           
        #image_crop = self._resize_norm(image_crop, self.image_size)        
       
        if random.random() > 0.5:
            image_obs_r = torch.flip(image_obs, [2])
            image_crop_r = torch.flip(image_crop, [2])
            cur_image_large_r = torch.flip(cur_image_large, [2])
            goal_image_full_r = torch.flip(goal_image_full, [2])
            goal_image_full_8_r = torch.flip(goal_image_full_8, [2])            
            goal_pose_r = np.array([relative_y/metric_waypoint_spacing, relative_x/metric_waypoint_spacing, np.cos(-goal_compass+cur_compass), np.sin(-goal_compass+cur_compass)])      

            PIL_current_crop_r = PIL_current_crop.transpose(Image.FLIP_LEFT_RIGHT)            
            PIL_goal_crop_r = PIL_goal_crop.transpose(Image.FLIP_LEFT_RIGHT)               
        else:
            image_obs_r = image_obs
            image_crop_r = image_crop
            cur_image_large_r = cur_image_large
            goal_image_full_r = goal_image_full
            goal_image_full_8_r = goal_image_full_8            
            goal_pose_r = np.array([relative_y/metric_waypoint_spacing, -relative_x/metric_waypoint_spacing, np.cos(goal_compass-cur_compass), np.sin(goal_compass-cur_compass)])    
            
            PIL_current_crop_r = PIL_current_crop
            PIL_goal_crop_r = PIL_goal_crop            
                 
        #ob_pose_norm = ob_pose_r/self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
        action_mask = (True)
        
        goad_id_virtual = -1
        for ig in range(20):
            if 0.3333*ig < radius and radius < 0.3333*(ig + 1):
                goad_id_virtual = ig
        if goad_id_virtual == -1:
            goad_id_virtual = 20
        if goal_id > 1:
            goad_id_virtual = 100
        """                            
        return (
            torch.as_tensor(image_obs_r, dtype=torch.float32),
            torch.as_tensor(image_crop_r, dtype=torch.float32),
            torch.as_tensor(cur_image_large_r, dtype=torch.float32),
            torch.as_tensor(goal_pose_r, dtype=torch.float32),
            torch.as_tensor(goal_image_full_r, dtype=torch.float32),   
            torch.as_tensor(goal_image_full_8_r, dtype=torch.float32),                      
            torch.as_tensor(goad_id_virtual, dtype=torch.float32),  
            torch.as_tensor(action_mask, dtype=torch.float32), 
            curr_path,           
            goal_path, 
        )              
        """
        obj_pose_norm = np.array((0.0, 0.0)) #dummy obj pose
        
        #print(goal_pos.shape, goal_yaw.shape)
        goal_pose_cos_sin = goal_pose_r #Adapting ViNT style action commands (X, Y, cos, sin)            
        #print(goal_pose_cos_sin.shape)
         
        ### Adapting OpenVLA stle ###
        actions = mbra_traj #np.random.rand(8, 4) #TODO loading pickle files
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)
        try:
            #lang = "move toward " + "[" + str(relative_y) + ", " + str(relative_x) + ", " + str(goal_compass-cur_compass) + "] on the current robot coordinate [X, Y, Yaw]." #I am giving X, Y and yaw angle." #dummy language prompt        
            lang = "move toward " + "xxxxx" #dummy language prompt
        except:
            print(inst_obj_x) 
        conversation = [
            #{"from": "human", "value": f"What action should the robot take to {lang}?"},
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
        #print("check!!", labels.size(), input_ids.size())
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        
        #dummy_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        #dummy_PIL = Image.fromarray(dummy_array)
        #pixel_values = self.image_transform(dummy_PIL)
        #print("bdd", PIL_current_crop_r.size, PIL_goal_crop_r.size)
        pixel_values = self.image_transform(PIL_current_crop_r)
        pixel_values_wrist = self.image_transform(PIL_goal_crop_r)
                
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        dataset_name = "bdd"

        current_map_image = np.asarray(np.random.rand(3, 96, 96), dtype=np.float32)
        goal_map_image = np.asarray(np.random.rand(3, 96, 96), dtype=np.float32)   

        dummy_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        dummy_PIL = Image.fromarray(dummy_array)
        pixel_values_dummy = self.image_transform(dummy_PIL)
        
        #return_dict = dict(pixel_values=pixel_values, pixel_values_wrist=pixel_values_wrist, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=torch.as_tensor(actions), actions_nomad=torch.as_tensor(actions), goal_pose=goal_pose_cos_sin, obj_pose_norm=obj_pose_norm, img_PIL=PIL_current_crop_r, cur_image_crop = image_crop_r, cur_image = image_obs_r, cur_image_large = cur_image_large_r, goal_image_crop=goal_image_full_r, goal_image_8=goal_image_full_8_r, temp_dist=goad_id_virtual, cur_map_image=current_map_image, goal_map_image=goal_map_image, pixel_values_curmap=pixel_values_dummy, pixel_values_goalmap=pixel_values_dummy)        
        return_dict = dict(pixel_values=pixel_values, pixel_values_wrist=pixel_values_wrist, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=torch.as_tensor(actions), actions_nomad=torch.as_tensor(np.asarray(np.random.rand(8, ACTION_DIM), dtype=np.float32)), goal_pose=goal_pose_cos_sin, obj_pose_norm=obj_pose_norm, img_PIL=PIL_current_crop_r, cur_image_crop = image_crop_r, cur_image = image_obs_r, cur_image_large = cur_image_large_r, goal_image_crop=goal_image_full_r, goal_image_8=goal_image_full_8_r, temp_dist=goad_id_virtual, cur_map_image=current_map_image, goal_map_image=goal_map_image, pixel_values_curmap=pixel_values_dummy, pixel_values_goalmap=pixel_values_dummy)
        #print(return_dict)
        return return_dict                 
        
        
class BDD_Dataset_multi_annot(Dataset):
    def __init__(
        self,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        data_split_type: str,
        data_folder: str,
        #pickle_folder: str,        
        #data_image_folder: str,
        #data_pickle_folder: str,       
        #lan_solo: bool = False,          
        context_type: str = "temporal",
        normalize: bool = True,
        model_type: str = "MBRA",
        #backside: bool = False,
        aug_seq: bool = False,
        #only_front: bool = False,                     
    ):
        """
        Main LeLaN dataset class

        Args:
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            image_size (list): Image size 224 x 224
            waypoint_spacing (int): Spacing between waypoints
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            data_split_type (string): train or test 
            data_image_folder (string): path for image
            data_pickle_folder (string): path for pickle file containing object pose, prompts and so on      
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            normalize (bool): Whether to normalize the distances or actions            
            backside (str): whether to use the backsie image or not (basically we can use the backside only for Go Stanford 4 and SACSoN.
            aug_seq (str): whether to use the image before and after.
            only_front (str): whether to contrain the back side image or not 
        """
        self.data_split_folder = data_split_folder
        self.data_split_type = data_split_type
        #self.data_image_folder = data_image_folder
        #self.data_pickle_folder = data_pickle_folder    
        self.data_folder = data_folder   
        #self.pickle_folder = pickle_folder             
        self.image_size = image_size
        
        if model_type == "MBRA":
            self.image_size_clip = (128, 416)
        else:
            self.image_size_clip = (224, 224)
        self.waypoint_spacing = waypoint_spacing
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.normalize = normalize
        #self.backside = backside
        self.aug_seq = aug_seq
        self.dataset_name = dataset_name
        #self.only_front = only_front

        # load data/data_config.yaml
        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}        
        self._load_split_index()
        self._build_caches_front()
        #self.lan_solo = lan_solo
        
        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2   

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches_front()

    def _build_caches_front(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}_{self.data_split_type}.lmdb",
        )

        #self._get_augdata()
        
        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    print(cache_filename, len(self.image_path))
                    for num in range(len(self.image_path)): 
                        for ih in range(self.context_size):
                            with open(self.image_path[num][ih], "rb") as f:
                                txn.put(self.image_path[num][ih].encode(), f.read())                    
                        """                       
                        if os.path.getsize(self.pickle_path[num]) > 0: 
                            with open(self.image_path[num], "rb") as f:
                                txn.put(self.image_path[num].encode(), f.read())
                        else:
                            print(self.image_path[num])
                        """
        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]
        
    def _remove_values_from_list(self, A, B):
        return [item for item in A if item not in B]
            
    def _load_split_index(self):
        if self.dataset_name == "bdd":
            self.v_random = 0.2 #for random cropping
            self.h_random = 0.1 #for random cropping 
                        
            image_path = []
            gps_list = []   
                                     
            folder_lst = next(os.walk(self.data_folder))[1]

            if self.data_split_type == "train":
                folder_lst_dataset = folder_lst[0:len(folder_lst)-int(len(folder_lst)*0.0)]
                print("BDD dataset train seq. number", len(folder_lst_dataset))
            else:
                folder_lst_dataset = folder_lst[len(folder_lst)-int(len(folder_lst)*0.1):len(folder_lst)]
                print("BDD dataset test seq. number", len(folder_lst_dataset))
                                
            for folder in folder_lst_dataset:
                print(self.data_split_type, folder)
                with open(self.data_folder + "/" + folder + "/gps.json", 'r', encoding='utf-8') as f:
                    gps_data = json.load(f)   
                gps_list += gps_data                          
                #for data_img in gps_data:
                #    number_files = len(gps_data)
                number_files = len(gps_data)
                for num in range(number_files):
                    #print(gps_data[num]["img_id"])
                    image_path_hist = []                  
                    for ih in range(self.context_size):
                        if gps_data[num]["img_id"] - ih > 0:
                            image_path_hist.append(self.data_folder + "/" + folder + "/img/" + str(gps_data[num]["img_id"] - ih) + '.jpg')       
                        else:
                            image_path_hist.append(self.data_folder + "/" + folder + "/img/" + str(0) + '.jpg')                                                                  
                    image_path.append(image_path_hist)
                        
        self.image_path = image_path
        self.gps_list = gps_list
        sum_num = 0
        for ic in range(len(self.image_path)):
            sum_num += len(self.image_path[ic])
            #print(self.image_path[ic])
            if len(self.image_path[ic]) != 5:
                print(self.image_path[ic])
        
        print("check BDD", len(self.image_path), len(self.gps_list), len(self.image_path[0]), sum_num)

    def _load_image_front(self, path):
        image_path = path
        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())               
                image_bytes = bytes(image_buffer)             
            image_bytes = io.BytesIO(image_bytes)         
            return img_path_to_data_front(image_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load image {image_path}")

    def _resize_norm(self, image, size):
        return TF.resize(image, size)
        #return TF.resize(2.0*(image - 0.5), size)
    """    
    def _get_augdata(self, ):
        aug_data_list = []
        for num in range(len(self.pickle_path)):
            if os.path.getsize(self.pickle_path[num]) > 0:            
                with open(self.pickle_path[num], "rb") as f:
                    aug_data = pickle.load(f)
            else:
                print(self.pickle_path[num])
            aug_data_list.append(aug_data)
            
        self.aug_data_list = aug_data_list
    """
    def _transform_position(self, lat, lon, heading, X, Y, theta):
        """
        Compute new latitude, longitude, and heading after moving by (X, Y, theta)
        in the local coordinate system where:
        - X is forward (aligned with heading)
        - Y is left (perpendicular to heading)
        - theta is counterclockwise (CCW)
        """
        # Convert lat/lon to UTM
        easting, northing, zone_number, zone_letter = self._latlon_to_utm(lat, lon)

        # Convert heading from degrees to radians
        # heading_rad = np.radians(heading)
        heading_rad = heading
        #new_heading = (heading - theta) 
        new_heading = (heading + theta) 
    
        # Corrected transformation: X moves forward, Y moves left
        #delta_easting = np.sqrt(X**2 + Y**2) * np.sin(new_heading) 
        #delta_northing = np.sqrt(X**2 + Y**2) * np.cos(new_heading) 
        delta_northing = X*np.cos(heading_rad) - Y*np.sin(heading_rad)
        delta_easting = -X*np.sin(heading_rad) - Y*np.cos(heading_rad)
    
        # New position in UTM coordinates
        new_easting = easting + delta_easting
        new_northing = northing + delta_northing

        # Convert back to latitude and longitude
        new_lat, new_lon = self._utm_to_latlon(new_easting, new_northing, zone_number, zone_letter)

        # Update heading (subtract for CCW rotation)
        #new_heading = (heading - theta)  
        new_heading = (heading + theta) 
        
        return new_lat, new_lon, new_heading    

    def _calculate_relative_position(self, x_a, y_a, x_b, y_b):
        delta_x = x_b - x_a
        delta_y = y_b - y_a
        return delta_x, delta_y

    def _rotate_to_local_frame(self, delta_x, delta_y, heading_a_rad):  
        # Apply the rotation matrix for the local frame
        relative_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        relative_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return relative_x, relative_y
            
    def __len__(self) -> int:
        return len(self.image_path)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image 
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        flag_data = 0
        iv = i
        
        it = random.random()
        #flag_back = 0
        #
        #if self.backside and ib > 0.5:
        #    flag_back = 1
        
        #remove data without object in front of the robot
        while flag_data == 0:
            image_fullsize = self._load_image_front(self.image_path[iv][0])
            #flag_data_inner = 0
            
            context_image = [image_fullsize]        
            for ih in range(self.context_size):
                context_image.append(self._load_image_front(self.image_path[iv][ih]))             
            """
            for ih in range(self.context_size + 1):
                if context_image[ih] is None:
                    flag_data_inner = 1
                    iv = random.randint(0, len(self.image_path)-1)
            """
            #goal_id = random.randint(0,5)
            goal_id = 1
            cur_pos = self.gps_list[iv]                
            try:
                #print(f"Accessing image_path at index {iv + goal_id}")
                #print("goal path", self.image_path[iv + goal_id][0])
                #print("cur path", self.image_path[iv][0])
                goal_image_full = self._load_image_front(self.image_path[iv + goal_id][0])
                goal_image_full_8 = self._load_image_front(self.image_path[iv + 1][0])     
                
                curr_path = self.image_path[iv][0]   
                goal_path = self.image_path[iv + 1][0]           
                goal_pos = self.gps_list[iv + goal_id]
            except:
                #print(f"[EXCEPT] Caught exception: {e}")
                goal_id = 0
                goal_image_full = self._load_image_front(self.image_path[iv + goal_id][0])   
                goal_image_full_8 = goal_image_full
                goal_pos = self.gps_list[iv + goal_id]     
                curr_path = self.image_path[iv][0]   
                goal_path = self.image_path[iv + goal_id][0]   
                            
            cur_compass = -cur_pos["course"]/180.0*3.141592
            goal_compass = -goal_pos["course"]/180.0*3.141592
                
            cur_utm = utm.from_latlon(cur_pos["latitude"], cur_pos["longitude"])
            goal_utm = utm.from_latlon(goal_pos["latitude"], goal_pos["longitude"])
            delta_x, delta_y = self._calculate_relative_position(cur_utm[0], cur_utm[1], goal_utm[0], goal_utm[1])
            relative_x, relative_y = self._rotate_to_local_frame(delta_x, delta_y, cur_compass)
            radius = np.sqrt(relative_x**2 + relative_y**2)
            ## Clipping the goal pose with thres_dist
            #radius = np.sqrt(relative_x**2 + relative_y**2)
            #if radius > thres_dist:
            #    relative_x = relative_x/radius*thres_dist
            #    relative_y = relative_y/radius*thres_dist                 
            
            
            #if it > 0.5 and abs(goal_compass - cur_compass) < 3.141592*20.0/180.0:
            #    iv = random.randint(0, len(self.image_path)-1)
            #else:
            #    flag_data = 1
            
            curr_path_list = curr_path.split("/")
            goal_path_list = goal_path.split("/")            
            limit_radius = 3.0    
            #print(curr_path_list[-3], goal_path_list[-3])
            #if radius > 15.0 and curr_path_list[-3] != goal_path_list[-3]:
            #    print(radius, curr_path_list[-3], goal_path_list[-3])
            #if radius > 15.0 or curr_path_list[-3] != goal_path_list[-3]:
            #    iv = random.randint(0, len(self.image_path)-1)
            #    flag_data = 0
            #else:
            if True:
                #print(relative_x, relative_y)    
                if radius > limit_radius and abs(goal_compass - cur_compass) < 3.141592*30.0/180.0:
                    #print("before", relative_x, relative_y)                
                    relative_x = relative_x/radius*limit_radius
                    relative_y = relative_y/radius*limit_radius
                    #print("after", relative_x, relative_y)
            flag_data = 1
            
        voffset = int(224.0*self.v_random*random.random())
        hoffset = int(224.0*self.h_random*random.random())

        image_obs_list = [] 
        image_crop_list = [] 
        metric_waypoint_spacing = 0.25 #normalization 
        for ih in range(self.context_size + 1):
            image_crop_list.append(self._resize_norm(context_image[ih][:, voffset:224-voffset, hoffset:224-hoffset], self.image_size))  
            image_obs_list.append(self._resize_norm(context_image[ih], self.image_size))     
        goal_image_full = self._resize_norm(goal_image_full[:, voffset:224-voffset, hoffset:224-hoffset], self.image_size)        
        goal_image_full_8 = self._resize_norm(goal_image_full_8, self.image_size) 
        cur_image_large = self._resize_norm(context_image[0][:, voffset:224-voffset, hoffset:224-hoffset], self.image_size_clip)                 
                              
        image_obs = torch.cat(image_obs_list[::-1])      
        image_crop = torch.cat(image_crop_list[::-1])           
        #image_crop = self._resize_norm(image_crop, self.image_size)        
       
        if random.random() > 0.5:
            image_obs_r = torch.flip(image_obs, [2])
            image_crop_r = torch.flip(image_crop, [2])
            cur_image_large_r = torch.flip(cur_image_large, [2])
            goal_image_full_r = torch.flip(goal_image_full, [2])
            goal_image_full_8_r = torch.flip(goal_image_full_8, [2])            
            goal_pose_r = np.array([relative_y/metric_waypoint_spacing, relative_x/metric_waypoint_spacing, np.cos(-goal_compass+cur_compass), np.sin(-goal_compass+cur_compass)])      
        else:
            image_obs_r = image_obs
            image_crop_r = image_crop
            cur_image_large_r = cur_image_large
            goal_image_full_r = goal_image_full
            goal_image_full_8_r = goal_image_full_8            
            goal_pose_r = np.array([relative_y/metric_waypoint_spacing, -relative_x/metric_waypoint_spacing, np.cos(goal_compass-cur_compass), np.sin(goal_compass-cur_compass)])         
        #ob_pose_norm = ob_pose_r/self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
        action_mask = (True)
        
        goad_id_virtual = -1
        for ig in range(20):
            if 0.3333*ig < radius and radius < 0.3333*(ig + 1):
                goad_id_virtual = ig
        if goad_id_virtual == -1:
            goad_id_virtual = 20
        if goal_id > 1:
            goad_id_virtual = 100
                                    
        return (
            torch.as_tensor(image_obs_r, dtype=torch.float32),
            torch.as_tensor(image_crop_r, dtype=torch.float32),
            torch.as_tensor(cur_image_large_r, dtype=torch.float32),
            torch.as_tensor(goal_pose_r, dtype=torch.float32),
            torch.as_tensor(goal_image_full_r, dtype=torch.float32),   
            torch.as_tensor(goal_image_full_8_r, dtype=torch.float32),                      
            torch.as_tensor(goad_id_virtual, dtype=torch.float32),  
            torch.as_tensor(action_mask, dtype=torch.float32), 
            curr_path,           
            goal_path, 
        )                           
