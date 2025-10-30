import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import tqdm
import io
import random
import lmdb

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from vint_train.data.data_utils import (
    img_path_to_data,
    calculate_sin_cos,
    get_data_path,
    to_local_coords,
)

from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform


def resize_depth(img: Image.Image, image_resize_size: Tuple[int, int]):
    img = img.resize(image_resize_size)
    return TF.to_tensor(img)


def img_path_to_data_depth(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
    return resize_depth(Image.open(path), image_resize_size)


def trans_mat(pos: float | np.ndarray | torch.Tensor, yaw: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(yaw, torch.Tensor):
        return torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), pos[0]],
                [torch.sin(yaw), torch.cos(yaw), pos[1]],
                [torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)],
            ],
        ), yaw
    else:
        yaw_value = yaw[0] if yaw.ndim == 1 else yaw
        return np.array(
            [
                [np.cos(yaw_value), -np.sin(yaw_value), pos[0]],
                [np.sin(yaw_value), np.cos(yaw_value), pos[1]],
                [0.0, 0.0, 1.0],
            ],
        ), yaw_value


class GNM_Dataset(Dataset):
    def __init__(
        self,
        action_tokenizer: PreTrainedTokenizerBase,
        base_tokenizer: ActionTokenizer,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        min_dist_cat: int,
        max_dist_cat: int,
        min_action_distance: int,
        max_action_distance: int,
        negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        predict_stop_token: bool = True,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
        obs_type: str = "image",
    ):
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name

        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        self.distance_categories = list(range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing))
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.negative_mining = negative_mining
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance
        self.context_size = context_size
        assert context_type in {"temporal", "randomized", "randomized_temporal"}
        self.context_type = context_type
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize
        self.obs_type = obs_type

        with open(os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r") as f:
            all_data_config = yaml.safe_load(f)
        assert self.dataset_name in all_data_config
        dataset_names = sorted(list(all_data_config.keys()))
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}
        self._load_index()
        self._build_caches()

        self.num_action_params = 3 if self.learn_angle else 2
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
        self._build_caches()


    def _build_caches(self, use_tqdm: bool = True):
        cache_filename = os.path.join(self.data_split_folder, f"dataset_{self.dataset_name}.lmdb")
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)
        # build the LMDB file if missing (write once)
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(self.goals_index, disable=not use_tqdm, dynamic_ncols=True,
                                  desc=f"Building LMDB cache for {self.dataset_name}")
            import lmdb
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        image_path = get_data_path(self.data_folder, traj_name, time)
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())

        # DO NOT open the env here — only store the path
        self._image_cache_path = cache_filename
        self._image_cache = None

    def _build_index(self, use_tqdm: bool = False):
        samples_index, goals_index = [], []
        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            for goal_time in range(traj_len):
                goals_index.append((traj_name, goal_time))
            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1)
                samples_index.append((traj_name, curr_time, max_goal_distance))
        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        goal_offset = np.random.randint(0, max_goal_dist + 1)
        if goal_offset == 0:
            trajectory_name, goal_time = self._sample_negative()
            return trajectory_name, goal_time, True
        else:
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_index(self) -> None:
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except:
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _get_image_cache(self):
        # Opens LMDB lazily. When called inside a worker, this creates a worker-local env.
        if self._image_cache is None:
            self._image_cache = lmdb.open(
                self._image_cache_path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=2048
            )
        return self._image_cache

    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(self.data_folder, trajectory_name, time)
        try:
            env = self._get_image_cache()
            with env.begin() as txn:
                buf = txn.get(image_path.encode())
            if buf is None:
                # handle missing key gracefully
                print(f"LMDB missing key {image_path}")
                return None
            return img_path_to_data(io.BytesIO(buf), self.image_size)
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return None

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]
        goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]
        goal_yaw = traj_data["yaw"][min(goal_time, len(traj_data["position"]) - 1)]

        if len(np.array([goal_yaw]).shape) == 2:
            goal_yaw = goal_yaw[0]
        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)
        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
        goal_yaw_loc = goal_yaw - yaw[0]

        if self.learn_angle:
            yaw = yaw[1:] - yaw[0]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            actions = waypoints[1:]

        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            goal_pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing

        return actions, goal_pos, goal_yaw_loc

    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)
    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)

        # Load images
        context = []
        if self.context_type == "temporal":
            # sample the last self.context_size times from interval [0, curr_time)
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        obs_image = torch.cat([
            self._load_image(f, t) for f, t in context
        ])
        obs_images_list = torch.split(obs_image, 3, dim=0)
        cur_image_large = TF.resize(obs_images_list[-1], (224, 224))
        
        # Load goal image
        goal_image = self._load_image(f_goal, goal_time)
        goal_image_large = TF.resize(goal_image, (224, 224))
        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Compute actions
        actions, goal_pos, goal_yaw = self._compute_actions(curr_traj_data, curr_time, goal_time)
        
        # Compute distances
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"
        
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)
        
        action_mask = (
            (distance < self.max_action_distance) and
            (distance > self.min_action_distance) and
            (not goal_is_negative)
        )

        obj_pose_norm = np.array((0.0, 0.0)) #dummy obj pose
        
        goal_pos = np.array(goal_pos)        # Ensures goal_pos supports slicing like [0:1]
        goal_yaw = np.array([goal_yaw])   
        goal_pose_cos_sin = np.concatenate((goal_pos[0:1], goal_pos[1:2], np.cos(goal_yaw[0:1]), np.sin(goal_yaw[0:1])), axis=0) #Adapting ViNT style action commands (X, Y, cos, sin)           
         
        ### Adapting OpenVLA stle ###
        actions = actions_torch
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
        #print("check!!", labels.size(), input_ids.size())
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)   
        
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        dataset_name = "gnm"
        
        if random.random() > 0.5:
            pixel_values = self.image_transform(to_pil_image(cur_image_large))
            pixel_values_g = self.image_transform(to_pil_image(goal_image_large))         
            cur_image_large = cur_image_large         
            obs_image = obs_image
            goal_image = goal_image
            actions = actions
            goal_pose_cos_sin = goal_pose_cos_sin               
        else:
            pixel_values = self.image_transform(to_pil_image(cur_image_large).transpose(Image.FLIP_LEFT_RIGHT))
            pixel_values_g = self.image_transform(to_pil_image(goal_image_large).transpose(Image.FLIP_LEFT_RIGHT))         
            cur_image_large = torch.flip(cur_image_large, [2])
            obs_image = torch.flip(obs_image, [2])
            goal_image = torch.flip(goal_image, [2])
            actions[:,1] = -actions[:,1]
            actions[:,3] = -actions[:,3]   
            goal_pose_cos_sin[1] = -goal_pose_cos_sin[1]
            goal_pose_cos_sin[3] = -goal_pose_cos_sin[3]                     
        
        # Set the available modality id for each dataset 
        # 0:"satellite only", 1:"pose and satellite", 2:"satellite and image", 3:"all", 4:"pose only", 5:"pose and image", 6:"image only", 7:"language only", 8:"language and pose"        
        modality_list = [4, 5, 6]   
        if distance <= 20:
            modality_id = random.choice(modality_list)
        else:
            modality_id = random.choice(modality_list[0:2]) #tdisntace is long --> no image only

        #action select 1.0: raw action, 0.0: MBRA synthetic action            
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
            img_PIL=to_pil_image(cur_image_large),
            gimg_PIL=to_pil_image(goal_image),
            cur_image = obs_image, 
            goal_image_8=goal_image, 
            temp_dist=distance,
            lan_prompt="No language instruction"       
        ) 
         
