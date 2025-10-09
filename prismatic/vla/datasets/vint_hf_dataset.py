import sys
# appending a path
#sys.path.insert(0, '/home/noriaki/Documents/lerobot')
#sys.path.append('/home/noriaki/Documents/map_cache')
#sys.path.append('/home/noriaki/Documents/lerobot')
#sys.path.append('/mnt/ephemeral2/noriaki/lerobot')
#
#print(sys.path)

#map_link = "/home/noriaki/Documents/map_cache/map_tiles_satellite"
map_link = "/home/noriaki/map_cache/map_tiles_satellite"

from enum import Enum
import numpy as np
from typing import Tuple, Callable
from pathlib import Path
import einops
import zarr

import torch
import torch.utils.data
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

#from vint_train.data.data_utils import (
#    to_local_coords,
#)

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    load_previous_and_future_frames,
)
from lerobot.common.datasets.video_utils import load_from_videos
#from map_cache import MapTileCache

from typing import Iterator
import random
from tqdm import tqdm
import pickle
import utm
import math

from torch.utils.data import Dataset
from itertools import accumulate
import torchvision

from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Type

#from ViNT
def yaw_rotmat(yaw: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(yaw, torch.Tensor):
        return torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw)],
                [torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)],
                [torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)],
            ],
        )
    else:
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )
        
def trans_mat(pos: float | np.ndarray | torch.Tensor, yaw: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(yaw, torch.Tensor):
        return torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), pos[0]],
                [torch.sin(yaw), torch.cos(yaw), pos[1]],
                [torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)],
            ],
        )
    else:
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), pos[0]],
                [np.sin(yaw), np.cos(yaw), pos[1]],
                [0.0, 0.0, 1.0],
            ],
        )

def to_local_coords(
    positions: np.ndarray | torch.Tensor, curr_pos: np.ndarray | torch.Tensor, curr_yaw: float | np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos) @ rotmat
    
def to_local_coords_yaw(
    positions: np.ndarray | torch.Tensor, curr_pos: np.ndarray | torch.Tensor, curr_yaw: float | np.ndarray | torch.Tensor,  goal_yaw: float | np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    cur_mat = trans_mat(curr_pos, curr_yaw)
    goal_mat = trans_mat(positions[0], goal_yaw)    
    cur_mat_inv = torch.linalg.inv(cur_mat)
    relative_mat = torch.matmul(cur_mat_inv, goal_mat)

    return relative_mat
    
#from ViNT end    

class ActionFormat(Enum):
    WAYPOINT = 1
    WAYPOINT_ANGLE = 2
    LINEAR_ANGULAR = 3

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_str(s: str) -> "ActionFormat":
        return ActionFormat[s.upper()]

def load_pickle(
    dataset: zarr.Array,
    index: int,
    episode_data_index: dict[str, torch.Tensor],  
    delta_timestamps: dict[str, list[float]],      
    ) -> dict[torch.Tensor]:
    #
    ep_id = dataset["episode_index"][index].item()
    ep_data_id_from = episode_data_index["from"][ep_id].item()
    ep_data_id_to = episode_data_index["to"][ep_id].item()
    ep_data_ids = torch.arange(ep_data_id_from, ep_data_id_to, 1)    
    
    for key, delta_ts in delta_timestamps.items():
        current_ts = dataset["timestamp"][index]
        query_ts = current_ts + torch.tensor(delta_ts)
        ep_timestamps = torch.from_numpy(dataset["timestamp"][ep_data_id_from:ep_data_id_to]).float()
        dist = torch.cdist(query_ts[:, None], ep_timestamps[:, None], p=1)
        min_, argmin_ = dist.min(1)
        data_ids = ep_data_ids[argmin_].numpy() 
    return data_ids
                
def load_frames_zarr(
    dataset: zarr.Array,
    index: int,
    episode_data_index: dict[str, torch.Tensor],
    delta_timestamps: dict[str, list[float]],
    tolerance_s: float,
) -> dict[torch.Tensor]:
    # get indices of the frames associated to the episode, and their timestamps
    ep_id = dataset["episode_index"][index].item()
    ep_data_id_from = episode_data_index["from"][ep_id].item()
    ep_data_id_to = episode_data_index["to"][ep_id].item()
    ep_data_ids = torch.arange(ep_data_id_from, ep_data_id_to, 1)

    # load timestamps
    ep_timestamps = torch.from_numpy(dataset["timestamp"][ep_data_id_from:ep_data_id_to]).float()

    # we make the assumption that the timestamps are sorted
    ep_first_ts = ep_timestamps[0]
    ep_last_ts = ep_timestamps[-1]
    current_ts = dataset["timestamp"][index]

    item = {}
    
    #print(dataset.keys())

    for key, delta_ts in delta_timestamps.items():
        # if it is a video frame
        timestamp_key = f"{key}.timestamp"
        path_key = f"{key}.path"
        is_video = timestamp_key in dataset.keys() and path_key in dataset.keys()

        # get timestamps used as query to retrieve data of previous/future frames
        if delta_ts is None:
            if key in dataset.keys():
                item[key] = torch.from_numpy(np.asarray(dataset[key][index]))
            elif is_video:
                item[key] = [
                    {"path": dataset[path_key][i.item()], "timestamp": dataset[timestamp_key][i.item()]}
                    for i in ep_data_ids
                ]
            else:
                raise ValueError(f"Timestamp key {timestamp_key} not found in dataset")
        else:
            query_ts = current_ts + torch.tensor(delta_ts)

            # compute distances between each query timestamp and all timestamps of all the frames belonging to the episode                               
            dist = torch.cdist(query_ts[:, None], ep_timestamps[:, None], p=1)
            min_, argmin_ = dist.min(1)

            # TODO(rcadene): synchronize timestamps + interpolation if needed

            is_pad = min_ > tolerance_s
            #assert ((query_ts[is_pad] < ep_first_ts) | (ep_last_ts < query_ts[is_pad])).all(), (
            #    f"One or several timestamps unexpectedly violate the tolerance ({min_} > {tolerance_s=}) inside episode range."
            #    "This might be due to synchronization issues with timestamps during data collection."
            #)

            # get dataset indices corresponding to frames to be loaded
            data_ids = ep_data_ids[argmin_].numpy()

            if is_video:
                # video mode where frame are expressed as dict of path and timestamp
                item[key] = [
                    {"path": dataset[path_key][i], "timestamp": float(dataset[timestamp_key][i])}
                    for i in data_ids
                ]
            else:
                item[key] = torch.from_numpy(dataset[key][data_ids])

            item[f"{key}_is_pad"] = is_pad

    return item

def load_from_videos_10k(
    item: dict[str, torch.Tensor],
    video_frame_keys: list[str],
    videos_dir: Path,
    tolerance_s: float,
    backend: str = "pyav",
):
    """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
    in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a Segmentation Fault.
    This probably happens because a memory reference to the video loader is created in the main process and a
    subprocess fails to access it.
    """
    # since video path already contains "videos" (e.g. videos_dir="data/videos", path="videos/episode_0.mp4")
    data_dir = videos_dir.parent

    for key in video_frame_keys:
        if isinstance(item[key], list):
            # load multiple frames at once (expected when delta_timestamps is not None)
            timestamps = [frame["timestamp"] for frame in item[key]]
            paths = [frame["path"] for frame in item[key]]
            if len(set(paths)) > 1:
                raise NotImplementedError("All video paths are expected to be the same for now.")
            video_path = data_dir / paths[0]
            #print("load_from_videos a", video_path)
            
            frames = decode_video_frames_torchvision_10k(video_path, timestamps, tolerance_s, backend)
            item[key] = frames
        else:
            # load one frame
            timestamps = [item[key]["timestamp"]]
            video_path = data_dir / item[key]["path"]
            #print("load_from_videos b", video_path)

            frames = decode_video_frames_torchvision_10k(video_path, timestamps, tolerance_s, backend)
            item[key] = frames[0]

    return item

def decode_video_frames_torchvision_10k(
    video_path: str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesnt support accuracte seepytho
    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    
    reader = torchvision.io.VideoReader(video_path, "video")
    
    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = timestamps[0]
    last_ts = timestamps[-1]
    
    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usally smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None
    
    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255
    
    assert len(timestamps) == len(closest_frames)
    
    """
    del log_loaded_timestamps
    del backend
    del tolerance_s
    del timestamps
    del closest_ts
    del is_within_tol
    del min_
    del argmin_
    del dist
    del loaded_ts
    del query_ts
    del reader
    del current_ts
    del last_ts
    del loaded_frames    
    del video_path
    del closest_frames
    """
    #gc.collect()
    #return [0.0]
    return closest_frames

class ViNTLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",
        action_format: ActionFormat | str = ActionFormat.WAYPOINT,
        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        context_size: int = 5,
        context_spacing: int = 1,
        dataset_framerate: int = 10,
        image_size: Tuple[int, int] = (120, 160),
        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.context_size = context_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
            },
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3
        img = TF.resize(img, self.image_size)
        if flip:
            img = torch.flip(img, dims=(-1,))
        return img

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))                      
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __getitem__(self, idx):
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))

        # Add the goal to the list of delta timestamps
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(self.goal_horizon)
        ] + [goal_dist * self.dt * self.action_spacing]

        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5 
        image_obs = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][:-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        image_goal = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)

        image_current, image_raw = self._image_transforms_depth(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)

        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)               
            
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([]) 
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                if flip_tf:                
                    robot_local_list.append([robot_local[0,2], -robot_local[1,2]]) 
                else:
                    robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                                    
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    ped_local = torch.matmul(robot_local, trans_ped)
                    
                    if flip_tf:
                        ped_local_list.append([ped_local[0,2], -ped_local[1,2]])
                        ped_list_notrans.append([ped_list[istep][1], ped_list[istep][0]])     
                    else:
                        ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                        ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                                        
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]

            ped_local_torch = torch.tensor(ped_local_list)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0

            ped_local_slice = ped_local_torch[9:31:3]         
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
            
            
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]   
        heading = item["observation.filtered_heading"][:-1]        
        
        goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
        relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)
        
        if flip_tf:
            goal_pos_relative[1] *= -1
            goal_heading *= -1
            relative_mat[0,1] *= -1
            relative_mat[1,0] *= -1
            relative_mat[1,2] *= -1
        
        if flip_tf:                  
            future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)                   
            direction = torch.stack([torch.cos(-heading), torch.sin(-heading)], dim=-1)
            action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(-heading))), -1, 1) * 5           
            unnorm_position[:,1] *= -1            
            action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
            action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing               
            future_positions_unfiltered[:,1] *= -1   
        else:
            direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
            action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
            action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
            action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing        
            future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)

        which_dataset = 0

        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action, dtype=torch.float32),
            torch.as_tensor(goal_dist/3.0, dtype=torch.int64),
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(image_raw, dtype=torch.float32),     
            torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),                            
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)
        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class ViNTDataset_10k(Dataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",
        action_format: ActionFormat | str = ActionFormat.WAYPOINT,
        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        context_size: int = 5,
        context_spacing: int = 1,
        dataset_framerate: int = 10,
        image_size: Tuple[int, int] = (120, 160),
        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.context_size = context_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson
        print("root_init", root)
        """
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
            },
        )
        """
        self.image_transforms = image_transforms        
        self.tolerance_s = 1e-4
        #self.videos_dir = Path(repo_id) / "videos"
        self.videos_dir = Path("/mnt/ephemeral2/frodobots_v2_export/frodobots_dataset_large/videos")
        self.video_backend = "pyav"
        
        img_history_spacing = [i * context_spacing * self.dt for i in range(-context_size, 1)]  # get next image obs too 
        action_future_spacing = [i * action_spacing * self.dt for i in range(action_horizon)]
        self.delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": img_history_spacing,
                "action": action_future_spacing,                
            }

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

        self.total_length = self.dataset_cache["action"].shape[0]
        print("Dataset Cache Loaded", self.dataset_cache["action"].shape)

        # Compute Episode Data Index 
        self.episode_data_index = self.get_episode_data_index(self.dataset_cache["episode_index"])
        print("Episode Index Computed")

    def get_episode_data_index(self, episode_index: list[int]) -> dict[str, torch.Tensor]:
        episode_lengths = []
        current_episode = episode_index[0]
        count = 0

        # Compute Episode Lengths 
        for ep in episode_index:
            if ep == current_episode:
                count += 1
            else:
                episode_lengths.append(count)
                current_episode = ep
                count = 1
        episode_lengths.append(count)  # Append the last episode's length

        # Compute Cumulative Lengths
        cumulative_lengths = list(accumulate(episode_lengths))
        return {
            "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
            "to": torch.LongTensor(cumulative_lengths),
        }

    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3
        img = TF.resize(img, self.image_size)
        if flip:
            img = torch.flip(img, dims=(-1,))
        return img

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))                      
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))

        # Add the goal to the list of delta timestamps
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(self.goal_horizon)
        ] + [goal_dist * self.dt * self.action_spacing]

        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5 
        image_obs = self._image_transforms(load_from_videos_10k(
            {"observation.images.front": item["observation.images.front"][:-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        image_goal = self._image_transforms(load_from_videos_10k(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)

        image_current, image_raw = self._image_transforms_depth(load_from_videos_10k(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)

        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)               
            
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([]) 
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                if flip_tf:                
                    robot_local_list.append([robot_local[0,2], -robot_local[1,2]]) 
                else:
                    robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                                    
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    ped_local = torch.matmul(robot_local, trans_ped)
                    
                    if flip_tf:
                        ped_local_list.append([ped_local[0,2], -ped_local[1,2]])
                        ped_list_notrans.append([ped_list[istep][1], ped_list[istep][0]])     
                    else:
                        ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                        ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                                        
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]

            ped_local_torch = torch.tensor(ped_local_list)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0

            ped_local_slice = ped_local_torch[9:31:3]         
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
            
            
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]   
        heading = item["observation.filtered_heading"][:-1]        
        
        goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
        relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)
        
        if flip_tf:
            goal_pos_relative[1] *= -1
            goal_heading *= -1
            relative_mat[0,1] *= -1
            relative_mat[1,0] *= -1
            relative_mat[1,2] *= -1
        
        if flip_tf:                  
            future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)                   
            direction = torch.stack([torch.cos(-heading), torch.sin(-heading)], dim=-1)
            action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(-heading))), -1, 1) * 5           
            unnorm_position[:,1] *= -1            
            action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
            action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing               
            future_positions_unfiltered[:,1] *= -1   
        else:
            direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
            action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
            action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
            action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing        
            future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)

        which_dataset = 0

        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action, dtype=torch.float32),
            torch.as_tensor(goal_dist/3.0, dtype=torch.int64),
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(image_raw, dtype=torch.float32),     
            torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),                            
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)
        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class ViNTLeRobotDataset_IL2_gps(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",
        action_format: ActionFormat | str = ActionFormat.WAYPOINT,
        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        goal_horizon2: int = 20,        
        context_size: int = 5,
        context_spacing: int = 1,
        dataset_framerate: int = 10,
        image_size: Tuple[int, int] = (120, 160),
        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.goal_horizon2 = goal_horizon2        
        self.context_size = context_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
            },
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }
        self.min_action_distance = 3
        self.max_action_distance = 20
        
    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #               
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))                   
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actionsViNTLeRobotDataset_IL2
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __getitem__(self, idx):
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist2 = min(8, episode_length_remaining)
        goal_dist_gps = np.random.randint(0, min(self.goal_horizon2, episode_length_remaining))    

        # Add the goal to the list of delta timestamps
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] + [goal_dist2 * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        control_horizon = 8+1
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing]
        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5     
        image_obs = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][:-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        image_goal2 = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf) #for inverse dynamics model
                
        image_goal = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)        
        
        image_current, image_raw = self._image_transforms_depth(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-4]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)                          
                
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)     
                    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([])    
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    ped_local = torch.matmul(robot_local, trans_ped)
                    ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                    ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                       
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]
       
            ped_local_torch = torch.tensor(ped_local_list)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0

            ped_local_slice = ped_local_torch[9:31:3]
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
         
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]
        
        if flip_tf:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]        
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)       
            goal_pos_relative[1] *= -1                
            relative_mat[0,1] *= -1
            relative_mat[1,0] *= -1                                    
        else:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)        
         
        action_IL = []
        metric_waypoint_spacing = 0.25 #normalization        
        goal_pos_relative = goal_pos_relative/metric_waypoint_spacing #normalization        
        for i_traj in range(control_horizon-1):
            traj_relative_mat = to_local_coords_yaw(item["observation.filtered_position"][i_traj + 1, None], unnorm_position[0], current_heading, item["observation.filtered_heading"][i_traj + 1])    
            
            if flip_tf:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, -traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], -traj_relative_mat[1,0]])
            else:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], traj_relative_mat[1,0]])
                            
        action_IL = torch.tensor(action_IL)        
        heading = item["observation.filtered_heading"][:-1]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        
        action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
        action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
        action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing

        future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)
        which_dataset = 0
        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")

        if goal_dist == 0:
            goal_is_negative = True
        else:
            goal_is_negative = False
        #[TODO] in GNM dataset, we set goal_dist == 20 when goal_dist is zero (current frame == goal frame) and we give random goal image. But we do in training code. We need to fix this not to confuse the users.
        
        action_mask = (
            (goal_dist < self.max_action_distance) and
            (goal_dist > self.min_action_distance) and
            (not goal_is_negative)
        )
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_goal2, dtype=torch.float32),            
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action_IL, dtype=torch.float32),
            torch.as_tensor(goal_dist, dtype=torch.int64),
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(action_mask, dtype=torch.float32),           
            torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)
        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class ViNTDataset_IL2_gps_10k(Dataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",
        action_format: ActionFormat | str = ActionFormat.WAYPOINT,
        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        goal_horizon2: int = 20,        
        context_size: int = 5,
        context_spacing: int = 1,
        dataset_framerate: int = 10,
        image_size: Tuple[int, int] = (120, 160),
        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.goal_horizon2 = goal_horizon2        
        self.context_size = context_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        """
        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
            },
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }
        """
         
        self.min_action_distance = 3
        self.max_action_distance = 20

        self.image_transforms = image_transforms        
        self.tolerance_s = 1e-4
        #self.videos_dir = Path(repo_id) / "videos"
        self.videos_dir = Path("/mnt/ephemeral2/frodobots_v2_export/frodobots_dataset/videos")
        self.video_backend = "pyav"
        
        img_history_spacing = [i * context_spacing * self.dt for i in range(-context_size, 1)]  # get next image obs too 
        action_future_spacing = [i * action_spacing * self.dt for i in range(action_horizon)]
        self.delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": img_history_spacing,
                "action": action_future_spacing,                
            }

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

        self.total_length = self.dataset_cache["action"].shape[0]
        print("Dataset Cache Loaded", self.dataset_cache["action"].shape)

        # Compute Episode Data Index 
        self.episode_data_index = self.get_episode_data_index(self.dataset_cache["episode_index"])
        print("Episode Index Computed")

    def get_episode_data_index(self, episode_index: list[int]) -> dict[str, torch.Tensor]:
        episode_lengths = []
        current_episode = episode_index[0]
        count = 0

        # Compute Episode Lengths 
        for ep in episode_index:
            if ep == current_episode:
                count += 1
            else:
                episode_lengths.append(count)
                current_episode = ep
                count = 1
        episode_lengths.append(count)  # Append the last episode's length

        # Compute Cumulative Lengths
        cumulative_lengths = list(accumulate(episode_lengths))
        return {
            "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
            "to": torch.LongTensor(cumulative_lengths),
        }
        
    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img

    def _image_transforms_rand_crop(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        
        #print(img.size())
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3

        #print(img.size())
        
        """
        if original_width / original_height > target_aspect:
            target_width = int(original_height * target_aspect)
            crop_left = (original_width - target_width) // 2
            print(crop_left, original_height, target_width)
            img = TF.resized_crop(img, 0, crop_left, original_height, target_width, self.image_size)
        else:
            target_height = int(original_width / target_aspect)
            crop_top = (original_height - target_height) // 2
            #print(target_height, crop_top)
            img = TF.resized_crop(img, crop_top, 0, target_height, original_width, self.image_size)
        """
        top = np.random.randint(0, 20)
        height = 144 - np.random.randint(0, 20) - top 
        left = np.random.randint(0, 25) 
        width = 256 - 2*left
        
        img = TF.crop(img, int(top), int(left), int(height), int(width))
        #print(img.size(), top, left, height, width)
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img     

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #               
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))                   
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actionsViNTLeRobotDataset_IL2
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist2 = min(8, episode_length_remaining)
        goal_dist_gps = np.random.randint(0, min(self.goal_horizon2, episode_length_remaining))    

        # Add the goal to the list of delta timestamps
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] + [goal_dist2 * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        control_horizon = 8+1
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing]
        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5     
        image_obs = self._image_transforms(load_from_videos_10k(
            {"observation.images.front": item["observation.images.front"][:-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        image_goal2 = self._image_transforms(load_from_videos_10k(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf) #for inverse dynamics model
                
        image_goal = self._image_transforms(load_from_videos_10k(
            {"observation.images.front": item["observation.images.front"][-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)        
        
        image_current, image_raw = self._image_transforms_depth(load_from_videos_10k(
            {"observation.images.front": item["observation.images.front"][-4]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)                          
                
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)     
                    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([])    
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    ped_local = torch.matmul(robot_local, trans_ped)
                    ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                    ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                       
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]
       
            ped_local_torch = torch.tensor(ped_local_list)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0

            ped_local_slice = ped_local_torch[9:31:3]
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
         
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]
        
        if flip_tf:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]        
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)       
            goal_pos_relative[1] *= -1                
            relative_mat[0,1] *= -1
            relative_mat[1,0] *= -1                                    
        else:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)        
         
        action_IL = []
        metric_waypoint_spacing = 0.25 #normalization        
        goal_pos_relative = goal_pos_relative/metric_waypoint_spacing #normalization        
        for i_traj in range(control_horizon-1):
            traj_relative_mat = to_local_coords_yaw(item["observation.filtered_position"][i_traj + 1, None], unnorm_position[0], current_heading, item["observation.filtered_heading"][i_traj + 1])    
            
            if flip_tf:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, -traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], -traj_relative_mat[1,0]])
            else:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], traj_relative_mat[1,0]])
                            
        action_IL = torch.tensor(action_IL)        
        heading = item["observation.filtered_heading"][:-1]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        
        action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
        action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
        action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing

        future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)
        which_dataset = 0
        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")

        if goal_dist == 0:
            goal_is_negative = True
        else:
            goal_is_negative = False
        #[TODO] in GNM dataset, we set goal_dist == 20 when goal_dist is zero (current frame == goal frame) and we give random goal image. But we do in training code. We need to fix this not to confuse the users.
        
        action_mask = (
            (goal_dist < self.max_action_distance) and
            (goal_dist > self.min_action_distance) and
            (not goal_is_negative)
        )
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_goal2, dtype=torch.float32),            
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action_IL, dtype=torch.float32),
            torch.as_tensor(goal_dist, dtype=torch.int64),
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(action_mask, dtype=torch.float32),           
            torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)
        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class ViNTDataset_IL2_gps_crop_10k(Dataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",
        action_format: ActionFormat | str = ActionFormat.WAYPOINT,
        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        goal_horizon2: int = 20,        
        context_size: int = 5,
        context_spacing: int = 1,
        dataset_framerate: int = 10,
        image_size: Tuple[int, int] = (120, 160),
        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.goal_horizon2 = goal_horizon2        
        self.context_size = context_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        """
        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
            },
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }
        """
         
        self.min_action_distance = 3
        self.max_action_distance = 20

        self.image_transforms = image_transforms        
        self.tolerance_s = 1e-4
        #self.videos_dir = Path(repo_id) / "videos"
        self.videos_dir = Path("/mnt/ephemeral2/frodobots_v2_export/frodobots_dataset/videos")
        self.video_backend = "pyav"
        
        img_history_spacing = [i * context_spacing * self.dt for i in range(-context_size, 1)]  # get next image obs too 
        action_future_spacing = [i * action_spacing * self.dt for i in range(action_horizon)]
        self.delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": img_history_spacing,
                "action": action_future_spacing,                
            }

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

        self.total_length = self.dataset_cache["action"].shape[0]
        print("Dataset Cache Loaded", self.dataset_cache["action"].shape)

        # Compute Episode Data Index 
        self.episode_data_index = self.get_episode_data_index(self.dataset_cache["episode_index"])
        print("Episode Index Computed")

    def get_episode_data_index(self, episode_index: list[int]) -> dict[str, torch.Tensor]:
        episode_lengths = []
        current_episode = episode_index[0]
        count = 0

        # Compute Episode Lengths 
        for ep in episode_index:
            if ep == current_episode:
                count += 1
            else:
                episode_lengths.append(count)
                current_episode = ep
                count = 1
        episode_lengths.append(count)  # Append the last episode's length

        # Compute Cumulative Lengths
        cumulative_lengths = list(accumulate(episode_lengths))
        return {
            "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
            "to": torch.LongTensor(cumulative_lengths),
        }
        
    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img

    def _image_transforms_rand_crop(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        
        #print(img.size())
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3

        #print(img.size())
        
        """
        if original_width / original_height > target_aspect:
            target_width = int(original_height * target_aspect)
            crop_left = (original_width - target_width) // 2
            print(crop_left, original_height, target_width)
            img = TF.resized_crop(img, 0, crop_left, original_height, target_width, self.image_size)
        else:
            target_height = int(original_width / target_aspect)
            crop_top = (original_height - target_height) // 2
            #print(target_height, crop_top)
            img = TF.resized_crop(img, crop_top, 0, target_height, original_width, self.image_size)
        """
        top = np.random.randint(0, 20)
        height = 144 - np.random.randint(0, 20) - top 
        left = np.random.randint(0, 25) 
        width = 256 - 2*left
        
        img = TF.crop(img, int(top), int(left), int(height), int(width))
        #print(img.size(), top, left, height, width)
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img     

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #               
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))                   
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actionsViNTLeRobotDataset_IL2
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist2 = min(8, episode_length_remaining)
        goal_dist_gps = np.random.randint(0, min(self.goal_horizon2, episode_length_remaining))    

        # Add the goal to the list of delta timestamps
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] + [goal_dist2 * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        control_horizon = 8+1
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing]
        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5     
        #image_obs = self._image_transforms(load_from_videos_10k(
        #    {"observation.images.front": item["observation.images.front"][:-3]},
        #    ["observation.images.front"],
        #    self.videos_dir,
        #    self.tolerance_s,
        #    self.video_backend,
        #)["observation.images.front"], flip_tf)

        image_obs_raw = load_from_videos(
            {"observation.images.front": item["observation.images.front"][:-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"]
         
        image_obs = self._image_transforms(image_obs_raw, flip_tf)
        image_obs_crop = self._image_transforms_rand_crop(image_obs_raw, flip_tf)
        
        image_goal2 = self._image_transforms(load_from_videos_10k(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf) #for inverse dynamics model
                
        image_goal = self._image_transforms_rand_crop(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)     
        
        image_current, image_raw = self._image_transforms_depth(load_from_videos_10k(
            {"observation.images.front": item["observation.images.front"][-4]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)                          
                
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)     
                    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([])    
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    ped_local = torch.matmul(robot_local, trans_ped)
                    ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                    ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                       
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]
       
            ped_local_torch = torch.tensor(ped_local_list)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0

            ped_local_slice = ped_local_torch[9:31:3]
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
         
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]
        
        if flip_tf:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]        
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)       
            goal_pos_relative[1] *= -1                
            relative_mat[0,1] *= -1
            relative_mat[1,0] *= -1                                    
        else:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)        
         
        action_IL = []
        metric_waypoint_spacing = 0.25 #normalization        
        goal_pos_relative = goal_pos_relative/metric_waypoint_spacing #normalization        
        for i_traj in range(control_horizon-1):
            traj_relative_mat = to_local_coords_yaw(item["observation.filtered_position"][i_traj + 1, None], unnorm_position[0], current_heading, item["observation.filtered_heading"][i_traj + 1])    
            
            if flip_tf:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, -traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], -traj_relative_mat[1,0]])
            else:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], traj_relative_mat[1,0]])
                            
        action_IL = torch.tensor(action_IL)        
        heading = item["observation.filtered_heading"][:-1]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        
        action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
        action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
        action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing

        future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)
        which_dataset = 0
        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")
        image_flattened_crop = einops.rearrange(image_obs_crop, "... t c h w -> ... (t c) h w")
        
        if goal_dist == 0:
            goal_is_negative = True
        else:
            goal_is_negative = False
        #[TODO] in GNM dataset, we set goal_dist == 20 when goal_dist is zero (current frame == goal frame) and we give random goal image. But we do in training code. We need to fix this not to confuse the users.
        
        action_mask = (
            (goal_dist < self.max_action_distance) and
            (goal_dist > self.min_action_distance) and
            (not goal_is_negative)
        )
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_flattened_crop, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_goal2, dtype=torch.float32),            
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action_IL, dtype=torch.float32),
            torch.as_tensor(goal_dist, dtype=torch.int64),
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(action_mask, dtype=torch.float32),           
            torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)
        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class ViNTLeRobotDataset_IL2_gps_map(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",
        action_format: ActionFormat | str = ActionFormat.WAYPOINT,
        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        goal_horizon2: int = 20,        
        context_size: int = 5,
        context_spacing: int = 1,
        dataset_framerate: int = 10,
        image_size: Tuple[int, int] = (120, 160),
        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.goal_horizon2 = goal_horizon2        
        self.context_size = context_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.latitude": [0.0],
                "observation.longitude": [0.0],     
                "observation.compass_heading": [0.0],                                
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
            },
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")        
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

        self.min_action_distance = 3
        self.max_action_distance = 20
        self.map_image_gen = MapTileCache(map_link)

        self.transform_PIL_tensor = transforms.ToTensor()
        
    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        img_rsize = TF.resize(img, (128, 416)) #          
        
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __getitem__(self, idx):
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist2 = min(8, episode_length_remaining)
        goal_dist_gps = np.random.randint(0, min(self.goal_horizon2, episode_length_remaining))    

        # Add the goal to the list of delta timestamps
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] + [goal_dist2 * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        control_horizon = 8+1
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing]

        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5     
        image_obs = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][:-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        image_goal2 = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf) #for inverse dynamics model
                
        image_goal = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        image_current, image_raw = self._image_transforms_depth(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-4]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)                         
                
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)      
                    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([]) 
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    ped_local = torch.matmul(robot_local, trans_ped)
                    ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                    ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                       
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]
            
            ped_local_torch = torch.tensor(ped_local_list)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0
            
            ped_local_slice = ped_local_torch[9:31:3]      
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
     
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]

        current_lat = item["observation.latitude"][0]
        current_lon = item["observation.longitude"][0]
        current_compass = item["observation.compass_heading"][0]        
        current_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), flip_tf)

        goal_lat = item["observation.latitude"][-1]
        goal_lon = item["observation.longitude"][-1]
        goal_compass = item["observation.compass_heading"][-1]      
        goal_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(goal_lat, goal_lon, goal_compass)), flip_tf)
        if flip_tf:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]        
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)       
            goal_pos_relative[1] *= -1                
            relative_mat[0,1] *= -1
            relative_mat[1,0] *= -1                                    
        else:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)        
         
        action_IL = []
        metric_waypoint_spacing = 0.25 #normalization        
        goal_pos_relative = goal_pos_relative/metric_waypoint_spacing #normalization        
        for i_traj in range(control_horizon-1):
            traj_relative_mat = to_local_coords_yaw(item["observation.filtered_position"][i_traj + 1, None], unnorm_position[0], current_heading, item["observation.filtered_heading"][i_traj + 1])    
            
            if flip_tf:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, -traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], -traj_relative_mat[1,0]])
            else:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], traj_relative_mat[1,0]])
                            
        action_IL = torch.tensor(action_IL)        
        heading = item["observation.filtered_heading"][:-1]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        
        action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
        action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
        action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing
        future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)
        which_dataset = 0

        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")
        if goal_dist == 0:
            goal_is_negative = True
        else:
            goal_is_negative = False
        #[TODO] in GNM dataset, we set goal_dist == 20 when goal_dist is zero (current frame == goal frame) and we give random goal image. But we do in training code. We need to fix this not to confuse the users.
        
        action_mask = (
            (goal_dist < self.max_action_distance) and
            (goal_dist > self.min_action_distance) and
            (not goal_is_negative)
        )
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_goal2, dtype=torch.float32),            
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action_IL, dtype=torch.float32),
            torch.as_tensor(goal_dist, dtype=torch.int64),
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(action_mask, dtype=torch.float32),                
            torch.as_tensor(current_map_image, dtype=torch.float32),         
            torch.as_tensor(goal_map_image, dtype=torch.float32),                 
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)
        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2
        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class ViNTLeRobotDataset_IL2_gps_map_crop(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",

        action_format: ActionFormat | str = ActionFormat.WAYPOINT,

        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        goal_horizon2: int = 20,        
        context_size: int = 5,
        context_spacing: int = 1,

        dataset_framerate: int = 10,

        image_size: Tuple[int, int] = (120, 160),

        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.goal_horizon2 = goal_horizon2        
        self.context_size = context_size
        #self.ped_past_size = ped_past_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.latitude": [0.0],
                "observation.longitude": [0.0],     
                "observation.compass_heading": [0.0],                                
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                #"action": [i * self.dt for i in range(action_spacing * action_horizon)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
                #"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)],
            },
            #video_backend = "video_reader",
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        #for k, v in self.dataset_cache.items():
        #    print(k, v.shape) 
        
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

        #self.min_action_distance = 0
        #self.max_action_distance = 100
        self.min_action_distance = 3
        self.max_action_distance = 20
        self.map_image_gen = MapTileCache(map_link)

        self.transform_PIL_tensor = transforms.ToTensor()
        
    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3

        #print(img.size())ped_list_no_trans
        
        """
        if original_width / original_height > target_aspect:
            target_width = int(original_height * target_aspect)
            crop_left = (original_width - target_width) // 2
            print(crop_left, original_height, target_width)
            img = TF.resized_crop(img, 0, crop_left, original_height, target_width, self.image_size)
        else:
            target_height = int(original_width / target_aspect)
            crop_top = (original_height - target_height) // 2
            #print(target_height, crop_top)
            img = TF.resized_crop(img, crop_top, 0, target_height, original_width, self.image_size)
        """
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img

    def _image_transforms_rand_crop(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        
        #print(img.size())
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3

        #print(img.size())
        
        """
        if original_width / original_height > target_aspect:
            target_width = int(original_height * target_aspect)
            crop_left = (original_width - target_width) // 2
            print(crop_left, original_height, target_width)
            img = TF.resized_crop(img, 0, crop_left, original_height, target_width, self.image_size)
        else:
            target_height = int(original_width / target_aspect)
            crop_top = (original_height - target_height) // 2
            #print(target_height, crop_top)
            img = TF.resized_crop(img, crop_top, 0, target_height, original_width, self.image_size)
        """
        top = np.random.randint(0, 20)
        height = 144 - np.random.randint(0, 20) - top 
        left = np.random.randint(0, 25) 
        width = 256 - 2*left
        
        img = TF.crop(img, int(top), int(left), int(height), int(width))
        #print(img.size(), top, left, height, width)
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img     

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #
        #img = TF.resize(img, (384, 512)) #for Zoedepth   
        
        """
        input_size = (616, 1064)
        h, w = img.shape[-2:]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        """  
        #img = TF.resize(img, (616, 1064)) #for metric3D            
        
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))
                      
        #print(img.size(), img_rsize.size())
                      
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __getitem__(self, idx):
        
        #print("data length", len(self.dataset_cache["episode_index"]))
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist2 = min(8, episode_length_remaining)
        goal_dist_gps = np.random.randint(0, min(self.goal_horizon2, episode_length_remaining))    
        #goal_dist_gps = min(self.goal_horizon2, episode_length_remaining)   
        #goal_dist = int(np.random.exponential(scale=self.goal_horizon)) #noriaki comment out
        #goal_dist = min(goal_dist, episode_length_remaining.item() - 1) #noriaki comment out

        # Add the goal to the list of delta timestamps
        #print("self", self.delta_timestamps["observation.images.front"], self.delta_timestamps["action"])
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] + [goal_dist2 * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        control_horizon = 8+1
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing]
        #delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
        #    i * self.dt * self.action_spacing
        #    for i in range(self.goal_horizon)
        #] + [goal_dist * self.dt * self.action_spacing]
        
        #print("check", delta_timestamps["observation.images.front"])ped_list_no_trans
        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5
        #print("observation.images.front", item["observation.images.front"][:])
        #print("observation.filtered_heading", item["observation.filtered_heading"][:])       
        
        image_obs_raw = load_from_videos(
            {"observation.images.front": item["observation.images.front"][:-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"]
         
        image_obs = self._image_transforms(image_obs_raw, flip_tf)
        image_obs_crop = self._image_transforms_rand_crop(image_obs_raw, flip_tf)
        """
        test = load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )
        """        
        image_goal2 = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf) #for inverse dynamics model
                
        image_goal = self._image_transforms_rand_crop(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        
        image_current, image_raw = self._image_transforms_depth(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-4]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        #self.map_image_gen
        
        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            #print("checking", item_pickle[30], delta_timestamps_sacson["pedestrian.filtered_position"][30], idx)
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)               
                #data = cloudpickle.load(file)               
                
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            #for id_ped in range(-30, 5):
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)    
                    #data = cloudpickle.load(file)    
                    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([])
            #print(ped_select, len(ped_list), ped_list)    
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    #print(trans_cur_inv, trans_fp, trans_ped)
                    ped_local = torch.matmul(robot_local, trans_ped)
                    ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                    ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                       
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)

            #print("value_list", value_list)
            #print("before", ped_local_list)  
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    #print(jj - value_list[ii], jj, value_list[ii]+jj)
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]
                    #ped_local_list[value_list[ii]+jj] = [ped_local_list[value_list[ii]][0] + (jj+1)*delta_p[0], ped_local_list[value_list[ii]][1] + (jj+1)*delta_p[1]]
            
            ped_local_torch = torch.tensor(ped_local_list)
            #ped_local_torch = ped_local_list
            #filtering
            #print("Before", ped_local_torch)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0
            #print("After", ped_local_torch) 
            
            #ped_local_slice = ped_local_list[30:7:-3][::-1] #3fps
            #print(ped_local_torch.size())
            #ped_local_torch = torch.flip(ped_local_torch, dims=[0])[3:30:3]
            ped_local_slice = ped_local_torch[9:31:3]
            #print(ped_local_slice.size(), ped_local_slice, ped_local_torch[30])
            #ped_local_slice = ped_local_torch[30:7:-3][::-1] #3fps            
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            #print(len(ped_local_slice), ped_local_list[30], ped_local_slice)
            #print(ped_local_list)
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
        """
        ped_list_no_trans = [0.0] #dummy
        ped_local_slice = [0.0] #dummy
        ped_local_slice_raw = [0.0] #dummy
        robot_local_slice = [0.0] #dummy
        
        """             
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]

        current_lat = item["observation.latitude"][0]
        current_lon = item["observation.longitude"][0]
        current_compass = item["observation.compass_heading"][0]        
        current_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), flip_tf)
        """
        image_PIL = self.map_image_gen.get_map_view(current_lat, current_lon, current_compass).resize((96, 96), resample=Image.Resampling.LANCZOS)
        image_noflip = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), False)
        image_flip = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), True)
        
        image_noflip_np = 255.0*image_noflip.permute(1, 2, 0).numpy()
        image_flip_np = 255.0*image_flip.permute(1, 2, 0).numpy()        
        #print(image_noflip.size(), image_flip.size())

        img_concat = np.clip(np.concatenate((image_PIL, image_noflip_np, image_flip_np), axis=1), 0, 255).astype(np.uint8)
        print(img_concat.shape)
        
        plt.imshow(img_concat)
        plt.axis('off')  # Hide the axis
        plt.savefig('saved_image.png')
        """
        goal_lat = item["observation.latitude"][-1]
        goal_lon = item["observation.longitude"][-1]
        goal_compass = item["observation.compass_heading"][-1]      
        goal_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(goal_lat, goal_lon, goal_compass)), flip_tf)
        #print("compass", current_compass, goal_compass)
        
        
        
                                        
        #print(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)
        if flip_tf:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]        
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)
            #print("before", goal_pos_relative, relative_mat)            
            goal_pos_relative[1] *= -1                
            relative_mat[0,1] *= -1
            relative_mat[1,0] *= -1                                    
            #print("after", goal_pos_relative, goal_dist_gps)
        else:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)        
         
        action_IL = []
        metric_waypoint_spacing = 0.25 #normalization        
        goal_pos_relative = goal_pos_relative/metric_waypoint_spacing #normalization        
        for i_traj in range(control_horizon-1):
            traj_relative_mat = to_local_coords_yaw(item["observation.filtered_position"][i_traj + 1, None], unnorm_position[0], current_heading, item["observation.filtered_heading"][i_traj + 1])    
            
            if flip_tf:
                #print("FLIP")
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, -traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], -traj_relative_mat[1,0]])
            else:
                #print("NO FLIP")
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], traj_relative_mat[1,0]])
                            
        action_IL = torch.tensor(action_IL)        
        #print(action_IL.size())
                
        # action = einops.reduce(item["action"][:-1], "(a s) d -> a d", reduction="mean", s=self.action_spacing)
        #action_is_pad = einops.reduce(item["action_is_pad"][:-1], "(a s) -> a", reduction="max", s=self.action_spacing)
        heading = item["observation.filtered_heading"][:-1]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        
        #print("action", heading, np.unwrap(heading), np.diff(np.unwrap(heading)))
        action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
        action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
        action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing
        
        #action = [0.0]
        
        future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)
        which_dataset = 0

        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")
        image_flattened_crop = einops.rearrange(image_obs_crop, "... t c h w -> ... (t c) h w")
        #image_flattened = [0.0]
        #image_goal = [0.0]
        #image_current = [0.0]
        #goal_dist = [0.0]
        #goal_pos_relative =                obs_image, 

        #relative_mat = [0.0]        
        #goal_heading = 0.0
        #current_heading = 0.0
        #which_dataset = [0.0]
        #future_positions_unfiltered = [0.0]
        #idx = [0.0]
        #image_raw = [0.0]

        if goal_dist == 0:
            goal_is_negative = True
        else:
            goal_is_negative = False
        #[TODO] in GNM dataset, we set goal_dist == 20 when goal_dist is zero (current frame == goal frame) and we give random goal image. But we do in training code. We need to fix this not to confuse the users.
        
        
        action_mask = (
            (goal_dist < self.max_action_distance) and
            (goal_dist > self.min_action_distance) and
            (not goal_is_negative)
        )
        #print(action_mask, goal_is_negative, goal_dist, self.max_action_distance, self.min_action_distance)
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_flattened_crop, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_goal2, dtype=torch.float32),            
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action_IL, dtype=torch.float32),
            #torch.as_tensor(goal_dist, dtype=torch.int64),
            torch.as_tensor(goal_dist_gps, dtype=torch.int64),            
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            #torch.as_tensor(~action_is_pad, dtype=torch.float32),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(action_mask, dtype=torch.float32),           
            #torch.as_tensor(image_raw, dtype=torch.float32),     
            #torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            #torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(current_map_image, dtype=torch.float32),         
            torch.as_tensor(goal_map_image, dtype=torch.float32),                 
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)

        # heading = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        # initial_heading = heading[indices]
        # final_heading = heading[target_indices]

        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class ViNTLeRobotDataset_IL2_gps_map2_crop(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",

        action_format: ActionFormat | str = ActionFormat.WAYPOINT,

        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        goal_horizon2: int = 20,        
        context_size: int = 5,
        context_spacing: int = 1,

        dataset_framerate: int = 10,

        image_size: Tuple[int, int] = (120, 160),

        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.goal_horizon2 = goal_horizon2        
        self.context_size = context_size
        #self.ped_past_size = ped_past_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.latitude": [0.0],
                "observation.longitude": [0.0],     
                "observation.compass_heading": [0.0],         
                "observation.utm_position": [0.0],
                "observation.utm_zone_letter": [0.0],     
                "observation.utm_zone_number": [0.0],                                           
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                #"action": [i * self.dt for i in range(action_spacing * action_horizon)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
                #"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)],
            },
            #video_backend = "video_reader",
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        #for k, v in self.dataset_cache.items():
        #    print(k, v.shape) 
        
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

        #self.min_action_distance = 0
        #self.max_action_distance = 100
        self.min_action_distance = 3
        self.max_action_distance = 20
        self.map_image_gen = MapTileCache(map_link)

        self.transform_PIL_tensor = transforms.ToTensor()
        
    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3

        #print(img.size())ped_list_no_trans
        
        """
        if original_width / original_height > target_aspect:
            target_width = int(original_height * target_aspect)
            crop_left = (original_width - target_width) // 2
            print(crop_left, original_height, target_width)
            img = TF.resized_crop(img, 0, crop_left, original_height, target_width, self.image_size)
        else:
            target_height = int(original_width / target_aspect)
            crop_top = (original_height - target_height) // 2
            #print(target_height, crop_top)
            img = TF.resized_crop(img, crop_top, 0, target_height, original_width, self.image_size)
        """
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img

    def _image_transforms_rand_crop(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        
        #print(img.size())
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3

        #print(img.size())
        
        """
        if original_width / original_height > target_aspect:
            target_width = int(original_height * target_aspect)
            crop_left = (original_width - target_width) // 2
            print(crop_left, original_height, target_width)
            img = TF.resized_crop(img, 0, crop_left, original_height, target_width, self.image_size)
        else:
            target_height = int(original_width / target_aspect)
            crop_top = (original_height - target_height) // 2
            #print(target_height, crop_top)
            img = TF.resized_crop(img, crop_top, 0, target_height, original_width, self.image_size)
        """
        top = np.random.randint(0, 20)
        height = 144 - np.random.randint(0, 20) - top 
        left = np.random.randint(0, 25) 
        width = 256 - 2*left
        
        img = TF.crop(img, int(top), int(left), int(height), int(width))
        #print(img.size(), top, left, height, width)
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img     

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #
        #img = TF.resize(img, (384, 512)) #for Zoedepth   
        
        """
        input_size = (616, 1064)
        h, w = img.shape[-2:]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        """  
        #img = TF.resize(img, (616, 1064)) #for metric3D            
        
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))
                      
        #print(img.size(), img_rsize.size())
                      
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def _latlon_to_utm(self, lat, lon):
        """ Convert latitude and longitude to UTM coordinates. """
        easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
        return easting, northing, zone_number, zone_letter

    def _utm_to_latlon(self, easting, northing, zone_number, zone_letter):
        """ Convert UTM coordinates back to latitude and longitude. """
        lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)
        return lat, lon

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

    def __getitem__(self, idx):
        
        #print("data length", len(self.dataset_cache["episode_index"]))
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist2 = min(8, episode_length_remaining)
        goal_dist_gps = np.random.randint(0, min(self.goal_horizon2, episode_length_remaining))    
        #goal_dist_gps = min(self.goal_horizon2, episode_length_remaining)   
        #goal_dist = int(np.random.exponential(scale=self.goal_horizon)) #noriaki comment out
        #goal_dist = min(goal_dist, episode_length_remaining.item() - 1) #noriaki comment out

        # Add the goal to the list of delta timestamps
        #print("self", self.delta_timestamps["observation.images.front"], self.delta_timestamps["action"])
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] + [goal_dist2 * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        control_horizon = 8+1
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing]
        #delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
        #    i * self.dt * self.action_spacing
        #    for i in range(self.goal_horizon)
        #] + [goal_dist * self.dt * self.action_spacing]
        
        #print("check", delta_timestamps["observation.images.front"])ped_list_no_trans
        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5
        #print("observation.images.front", item["observation.images.front"][:])
        #print("observation.filtered_heading", item["observation.filtered_heading"][:])       
        
        image_obs_raw = load_from_videos(
            {"observation.images.front": item["observation.images.front"][:-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"]
         
        image_obs = self._image_transforms(image_obs_raw, flip_tf)
        image_obs_crop = self._image_transforms_rand_crop(image_obs_raw, flip_tf)
        """
        test = load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )
        """        
        image_goal2 = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf) #for inverse dynamics model
                
        image_goal = self._image_transforms_rand_crop(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        
        image_current, image_raw = self._image_transforms_depth(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-4]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        #self.map_image_gen
        
        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            #print("checking", item_pickle[30], delta_timestamps_sacson["pedestrian.filtered_position"][30], idx)
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)               
                #data = cloudpickle.load(file)               
                
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            #for id_ped in range(-30, 5):
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)    
                    #data = cloudpickle.load(file)    
                    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([])
            #print(ped_select, len(ped_list), ped_list)    
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    #print(trans_cur_inv, trans_fp, trans_ped)
                    ped_local = torch.matmul(robot_local, trans_ped)
                    ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                    ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                       
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)

            #print("value_list", value_list)
            #print("before", ped_local_list)  
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    #print(jj - value_list[ii], jj, value_list[ii]+jj)
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]
                    #ped_local_list[value_list[ii]+jj] = [ped_local_list[value_list[ii]][0] + (jj+1)*delta_p[0], ped_local_list[value_list[ii]][1] + (jj+1)*delta_p[1]]
            
            ped_local_torch = torch.tensor(ped_local_list)
            #ped_local_torch = ped_local_list
            #filtering
            #print("Before", ped_local_torch)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0
            #print("After", ped_local_torch) 
            
            #ped_local_slice = ped_local_list[30:7:-3][::-1] #3fps
            #print(ped_local_torch.size())
            #ped_local_torch = torch.flip(ped_local_torch, dims=[0])[3:30:3]
            ped_local_slice = ped_local_torch[9:31:3]
            #print(ped_local_slice.size(), ped_local_slice, ped_local_torch[30])
            #ped_local_slice = ped_local_torch[30:7:-3][::-1] #3fps            
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            #print(len(ped_local_slice), ped_local_list[30], ped_local_slice)
            #print(ped_local_list)
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
        """
        ped_list_no_trans = [0.0] #dummy
        ped_local_slice = [0.0] #dummy
        ped_local_slice_raw = [0.0] #dummy
        robot_local_slice = [0.0] #dummy
        
        """             
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]

        current_lat = item["observation.latitude"][0]
        current_lon = item["observation.longitude"][0]
        #current_compass = item["observation.compass_heading"][0]  
        current_compass = item["observation.filtered_heading"][0]       
        current_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), flip_tf)
        """
        image_PIL = self.map_image_gen.get_map_view(current_lat, current_lon, current_compass).resize((96, 96), resample=Image.Resampling.LANCZOS)
        image_noflip = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), False)
        image_flip = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), True)
        
        image_noflip_np = 255.0*image_noflip.permute(1, 2, 0).numpy()
        image_flip_np = 255.0*image_flip.permute(1, 2, 0).numpy()        
        #print(image_noflip.size(), image_flip.size())

        img_concat = np.clip(np.concatenate((image_PIL, image_noflip_np, image_flip_np), axis=1), 0, 255).astype(np.uint8)
        print(img_concat.shape)
        
        plt.imshow(img_concat)
        plt.axis('off')  # Hide the axis
        plt.savefig('saved_image.png')
        """
        #goal_lat = item["observation.latitude"][-1]
        #goal_lon = item["observation.longitude"][-1]
        #goal_compass = item["observation.compass_heading"][-1]      
        #goal_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(goal_lat, goal_lon, goal_compass)), flip_tf)
        #print("compass", current_compass, goal_compass)    
        
                                        
        #print(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)
        if flip_tf:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]        
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)
            
            x_loc = relative_mat[0,2]
            y_loc = relative_mat[1,2]
            yaw_loc = np.arctan2(relative_mat[1,0], relative_mat[1,1])
            #print(yaw_loc, goal_heading-current_heading)            
            #filt_goal = item["observation.filtered_position"][-1, None]
            #utm_goal = item["observation.utm_position"][-1, None]
            #let_goal = item["observation.utm_zone_letter"][-1, None]
            #num_goal = item["observation.utm_zone_number"][-1, None]
            #print("dataset", filt_goal, utm_goal, let_goal, num_goal)
            
            new_lat, new_lon, new_heading = self._transform_position(current_lat.item(), current_lon.item(), current_compass.item(), x_loc.item(), y_loc.item(), yaw_loc.item())              
            try:
                goal_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(new_lat, new_lon, new_heading)), flip_tf)
            except Exception as e:
                goal_lat = item["observation.latitude"][-1]
                goal_lon = item["observation.longitude"][-1]
                #goal_compass = item["observation.compass_heading"][-1] 
                goal_compass = item["observation.filtered_heading"][-1]                      
                goal_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(goal_lat, goal_lon, goal_compass)), flip_tf)      
            
            
            #print("before", goal_pos_relative, relative_mat)            
            goal_pos_relative[1] *= -1                
            relative_mat[0,1] *= -1
            relative_mat[1,0] *= -1                                    
            #print("after", goal_pos_relative, goal_dist_gps)
        else:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)        

            x_loc = relative_mat[0,2]
            y_loc = relative_mat[1,2]
            yaw_loc = np.arctan2(relative_mat[1,0], relative_mat[1,1])
            #print(yaw_loc, goal_heading-current_heading)
            
            new_lat, new_lon, new_heading = self._transform_position(current_lat.item(), current_lon.item(), current_compass.item(), x_loc.item(), y_loc.item(), yaw_loc.item())        
            try:
                goal_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(new_lat, new_lon, new_heading)), flip_tf)
            except Exception as e:
                goal_lat = item["observation.latitude"][-1]
                goal_lon = item["observation.longitude"][-1]
                #goal_compass = item["observation.compass_heading"][-1] 
                goal_compass = item["observation.filtered_heading"][-1]                      
                goal_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(goal_lat, goal_lon, goal_compass)), flip_tf)     
                  
         
        action_IL = []
        metric_waypoint_spacing = 0.25 #normalization        
        goal_pos_relative = goal_pos_relative/metric_waypoint_spacing #normalization        
        for i_traj in range(control_horizon-1):
            traj_relative_mat = to_local_coords_yaw(item["observation.filtered_position"][i_traj + 1, None], unnorm_position[0], current_heading, item["observation.filtered_heading"][i_traj + 1])    
            
            if flip_tf:
                #print("FLIP")
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, -traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], -traj_relative_mat[1,0]])
            else:
                #print("NO FLIP")
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], traj_relative_mat[1,0]])
                            
        action_IL = torch.tensor(action_IL)        
        #print(action_IL.size())
                
        # action = einops.reduce(item["action"][:-1], "(a s) d -> a d", reduction="mean", s=self.action_spacing)
        #action_is_pad = einops.reduce(item["action_is_pad"][:-1], "(a s) -> a", reduction="max", s=self.action_spacing)
        heading = item["observation.filtered_heading"][:-1]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        
        #print("action", heading, np.unwrap(heading), np.diff(np.unwrap(heading)))
        action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
        action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
        action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing
        
        #action = [0.0]
        
        future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)
        which_dataset = 0

        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")
        image_flattened_crop = einops.rearrange(image_obs_crop, "... t c h w -> ... (t c) h w")
        #image_flattened = [0.0]
        #image_goal = [0.0]
        #image_current = [0.0]
        #goal_dist = [0.0]
        #goal_pos_relative =                obs_image, 

        #relative_mat = [0.0]        
        #goal_heading = 0.0
        #current_heading = 0.0
        #which_dataset = [0.0]
        #future_positions_unfiltered = [0.0]
        #idx = [0.0]
        #image_raw = [0.0]

        if goal_dist == 0:
            goal_is_negative = True
        else:
            goal_is_negative = False
        #[TODO] in GNM dataset, we set goal_dist == 20 when goal_dist is zero (current frame == goal frame) and we give random goal image. But we do in training code. We need to fix this not to confuse the users.
        
        
        action_mask = (
            (goal_dist < self.max_action_distance) and
            (goal_dist > self.min_action_distance) and
            (not goal_is_negative)
        )
        #print(action_mask, goal_is_negative, goal_dist, self.max_action_distance, self.min_action_distance)
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_flattened_crop, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_goal2, dtype=torch.float32),            
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action_IL, dtype=torch.float32),
            #torch.as_tensor(goal_dist, dtype=torch.int64),
            torch.as_tensor(goal_dist_gps, dtype=torch.int64),            
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            #torch.as_tensor(~action_is_pad, dtype=torch.float32),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(action_mask, dtype=torch.float32),           
            #torch.as_tensor(image_raw, dtype=torch.float32),     
            #torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            #torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(current_map_image, dtype=torch.float32),         
            torch.as_tensor(goal_map_image, dtype=torch.float32),                 
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)

        # heading = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        # initial_heading = heading[indices]
        # final_heading = heading[target_indices]

        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class ViNTLeRobotDataset_IL2_gps_map_crop_test(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",

        action_format: ActionFormat | str = ActionFormat.WAYPOINT,

        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        goal_horizon2: int = 20,        
        context_size: int = 5,
        context_spacing: int = 1,

        dataset_framerate: int = 10,

        image_size: Tuple[int, int] = (120, 160),

        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.goal_horizon2 = goal_horizon2        
        self.context_size = context_size
        #self.ped_past_size = ped_past_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.latitude": [0.0],
                "observation.longitude": [0.0],     
                "observation.compass_heading": [0.0],                                
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                #"action": [i * self.dt for i in range(action_spacing * action_horizon)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
                #"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)],
            },
            #video_backend = "video_reader",
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        #for k, v in self.dataset_cache.items():
        #    print(k, v.shape) 
        
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

        #self.min_action_distance = 0
        #self.max_action_distance = 100
        self.min_action_distance = 3
        self.max_action_distance = 20
        self.map_image_gen = MapTileCache(map_link)

        self.transform_PIL_tensor = transforms.ToTensor()
        
    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3

        #print(img.size())ped_list_no_trans
        
        """
        if original_width / original_height > target_aspect:
            target_width = int(original_height * target_aspect)
            crop_left = (original_width - target_width) // 2
            print(crop_left, original_height, target_width)
            img = TF.resized_crop(img, 0, crop_left, original_height, target_width, self.image_size)
        else:
            target_height = int(original_width / target_aspect)
            crop_top = (original_height - target_height) // 2
            #print(target_height, crop_top)
            img = TF.resized_crop(img, crop_top, 0, target_height, original_width, self.image_size)
        """
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img

    def _image_transforms_rand_crop(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        
        #print(img.size())
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3

        #print(img.size())
        
        """
        if original_width / original_height > target_aspect:
            target_width = int(original_height * target_aspect)
            crop_left = (original_width - target_width) // 2
            print(crop_left, original_height, target_width)
            img = TF.resized_crop(img, 0, crop_left, original_height, target_width, self.image_size)
        else:
            target_height = int(original_width / target_aspect)
            crop_top = (original_height - target_height) // 2
            #print(target_height, crop_top)
            img = TF.resized_crop(img, crop_top, 0, target_height, original_width, self.image_size)
        """
        top = np.random.randint(0, 20)
        height = 144 - np.random.randint(0, 20) - top 
        left = np.random.randint(0, 25) 
        width = 256 - 2*left
        
        img = TF.crop(img, int(top), int(left), int(height), int(width))
        #print(img.size(), top, left, height, width)
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img     

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #
        #img = TF.resize(img, (384, 512)) #for Zoedepth   
        
        """
        input_size = (616, 1064)
        h, w = img.shape[-2:]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        """  
        #img = TF.resize(img, (616, 1064)) #for metric3D            
        
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))
                      
        #print(img.size(), img_rsize.size())
                      
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __getitem__(self, idx):
        
        #print("data length", len(self.dataset_cache["episode_index"]))
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist2 = min(8, episode_length_remaining)
        goal_dist_gps = np.random.randint(0, min(self.goal_horizon2, episode_length_remaining))    
        #goal_dist_gps = min(self.goal_horizon2, episode_length_remaining)   
        #goal_dist = int(np.random.exponential(scale=self.goal_horizon)) #noriaki comment out
        #goal_dist = min(goal_dist, episode_length_remaining.item() - 1) #noriaki comment out

        # Add the goal to the list of delta timestamps
        #print("self", self.delta_timestamps["observation.images.front"], self.delta_timestamps["action"])
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] + [goal_dist2 * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        control_horizon = 8+1
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing]
        #delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
        #    i * self.dt * self.action_spacing
        #    for i in range(self.goal_horizon)
        #] + [goal_dist * self.dt * self.action_spacing]
        
        #print("check", delta_timestamps["observation.images.front"])ped_list_no_trans
        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        #flip_tf = random.random() > 0.5
        flip_tf = False
        #print("observation.images.front", item["observation.images.front"][:])
        #print("observation.filtered_heading", item["observation.filtered_heading"][:])       
        
        image_obs_raw = load_from_videos(
            {"observation.images.front": item["observation.images.front"][:-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"]
         
        image_obs = self._image_transforms(image_obs_raw, flip_tf)
        image_obs_crop = self._image_transforms_rand_crop(image_obs_raw, flip_tf)
        """
        test = load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )
        """        
        image_goal2 = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf) #for inverse dynamics model
                
        image_goal = self._image_transforms_rand_crop(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        
        image_current, image_raw = self._image_transforms_depth(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-4]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        #self.map_image_gen
        
        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            #print("checking", item_pickle[30], delta_timestamps_sacson["pedestrian.filtered_position"][30], idx)
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)               
                #data = cloudpickle.load(file)               
                
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            #for id_ped in range(-30, 5):
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)    
                    #data = cloudpickle.load(file)    
                    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([])
            #print(ped_select, len(ped_list), ped_list)    
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    #print(trans_cur_inv, trans_fp, trans_ped)
                    ped_local = torch.matmul(robot_local, trans_ped)
                    ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                    ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                       
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)

            #print("value_list", value_list)
            #print("before", ped_local_list)  
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    #print(jj - value_list[ii], jj, value_list[ii]+jj)
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]
                    #ped_local_list[value_list[ii]+jj] = [ped_local_list[value_list[ii]][0] + (jj+1)*delta_p[0], ped_local_list[value_list[ii]][1] + (jj+1)*delta_p[1]]
            
            ped_local_torch = torch.tensor(ped_local_list)
            #ped_local_torch = ped_local_list
            #filtering
            #print("Before", ped_local_torch)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0
            #print("After", ped_local_torch) 
            
            #ped_local_slice = ped_local_list[30:7:-3][::-1] #3fps
            #print(ped_local_torch.size())
            #ped_local_torch = torch.flip(ped_local_torch, dims=[0])[3:30:3]
            ped_local_slice = ped_local_torch[9:31:3]
            #print(ped_local_slice.size(), ped_local_slice, ped_local_torch[30])
            #ped_local_slice = ped_local_torch[30:7:-3][::-1] #3fps            
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            #print(len(ped_local_slice), ped_local_list[30], ped_local_slice)
            #print(ped_local_list)
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
        """
        ped_list_no_trans = [0.0] #dummy
        ped_local_slice = [0.0] #dummy
        ped_local_slice_raw = [0.0] #dummy
        robot_local_slice = [0.0] #dummy
        
        """             
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]

        current_lat = item["observation.latitude"][0]
        current_lon = item["observation.longitude"][0]
        current_compass = item["observation.compass_heading"][0]        
        current_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), flip_tf)
        """
        image_PIL = self.map_image_gen.get_map_view(current_lat, current_lon, current_compass).resize((96, 96), resample=Image.Resampling.LANCZOS)
        image_noflip = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), False)
        image_flip = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), True)
        
        image_noflip_np = 255.0*image_noflip.permute(1, 2, 0).numpy()
        image_flip_np = 255.0*image_flip.permute(1, 2, 0).numpy()        
        #print(image_noflip.size(), image_flip.size())

        img_concat = np.clip(np.concatenate((image_PIL, image_noflip_np, image_flip_np), axis=1), 0, 255).astype(np.uint8)
        print(img_concat.shape)
        
        plt.imshow(img_concat)
        plt.axis('off')  # Hide the axis
        plt.savefig('saved_image.png')
        """
        goal_lat = item["observation.latitude"][-1]
        goal_lon = item["observation.longitude"][-1]
        goal_compass = item["observation.compass_heading"][-1]      
        goal_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(goal_lat, goal_lon, goal_compass)), flip_tf)
        #print("compass", current_compass, goal_compass)
        
        
        
                                        
        #print(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)
        if flip_tf:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]        
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)
            #print("before", goal_pos_relative, relative_mat)            
            goal_pos_relative[1] *= -1                
            relative_mat[0,1] *= -1
            relative_mat[1,0] *= -1                                    
            #print("after", goal_pos_relative, goal_dist_gps)
        else:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)        
         
        action_IL = []
        metric_waypoint_spacing = 0.25 #normalization        
        goal_pos_relative = goal_pos_relative/metric_waypoint_spacing #normalization        
        for i_traj in range(control_horizon-1):
            traj_relative_mat = to_local_coords_yaw(item["observation.filtered_position"][i_traj + 1, None], unnorm_position[0], current_heading, item["observation.filtered_heading"][i_traj + 1])    
            
            if flip_tf:
                #print("FLIP")
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, -traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], -traj_relative_mat[1,0]])
            else:
                #print("NO FLIP")
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], traj_relative_mat[1,0]])
                            
        action_IL = torch.tensor(action_IL)        
        #print(action_IL.size())
                
        # action = einops.reduce(item["action"][:-1], "(a s) d -> a d", reduction="mean", s=self.action_spacing)
        #action_is_pad = einops.reduce(item["action_is_pad"][:-1], "(a s) -> a", reduction="max", s=self.action_spacing)
        heading = item["observation.filtered_heading"][:-1]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        
        #print("action", heading, np.unwrap(heading), np.diff(np.unwrap(heading)))
        action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
        action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
        action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing
        
        #action = [0.0]
        
        future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)
        which_dataset = 0

        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")
        image_flattened_crop = einops.rearrange(image_obs_crop, "... t c h w -> ... (t c) h w")
        #image_flattened = [0.0]
        #image_goal = [0.0]
        #image_current = [0.0]
        #goal_dist = [0.0]
        #goal_pos_relative =                obs_image, 

        #relative_mat = [0.0]        
        #goal_heading = 0.0
        #current_heading = 0.0
        #which_dataset = [0.0]
        #future_positions_unfiltered = [0.0]
        #idx = [0.0]
        #image_raw = [0.0]

        if goal_dist == 0:
            goal_is_negative = True
        else:
            goal_is_negative = False
        #[TODO] in GNM dataset, we set goal_dist == 20 when goal_dist is zero (current frame == goal frame) and we give random goal image. But we do in training code. We need to fix this not to confuse the users.
        
        
        action_mask = (
            (goal_dist < self.max_action_distance) and
            (goal_dist > self.min_action_distance) and
            (not goal_is_negative)
        )
        #print(action_mask, goal_is_negative, goal_dist, self.max_action_distance, self.min_action_distance)
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_flattened_crop, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_goal2, dtype=torch.float32),            
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action_IL, dtype=torch.float32),
            #torch.as_tensor(goal_dist, dtype=torch.int64),
            torch.as_tensor(goal_dist_gps, dtype=torch.int64),            
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            #torch.as_tensor(~action_is_pad, dtype=torch.float32),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(action_mask, dtype=torch.float32),           
            #torch.as_tensor(image_raw, dtype=torch.float32),     
            #torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            #torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(current_map_image, dtype=torch.float32),         
            torch.as_tensor(goal_map_image, dtype=torch.float32),                 
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),
            torch.as_tensor(robot_local_slice, dtype=torch.float32),    
            torch.as_tensor(goal_lat, dtype=torch.float32),                 
            torch.as_tensor(goal_lon, dtype=torch.float32),
            torch.as_tensor(goal_compass, dtype=torch.float32),              
            torch.as_tensor(current_lat, dtype=torch.float32),                 
            torch.as_tensor(current_lon, dtype=torch.float32),
            torch.as_tensor(current_compass, dtype=torch.float32),                                                     
        )  
  
    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)

        # heading = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        # initial_heading = heading[indices]
        # final_heading = heading[target_indices]

        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class ViNTLeRobotDataset_IL2(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",
        action_format: ActionFormat | str = ActionFormat.WAYPOINT,
        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        context_size: int = 5,
        context_spacing: int = 1,
        dataset_framerate: int = 10,
        image_size: Tuple[int, int] = (120, 160),
        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.context_size = context_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
            },
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")        
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

        self.min_action_distance = 3
        self.max_action_distance = 20
        
    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))
                                            
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __getitem__(self, idx):
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist2 = min(8, episode_length_remaining)

        # Add the goal to the list of delta timestamps
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing]  + [goal_dist2 * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        control_horizon = 8+1
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing]
        
        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5    
        image_obs = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][:-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        image_goal2 = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf) #for inverse dynamics model
                
        image_goal = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
           
        image_current, image_raw = self._image_transforms_depth(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        
        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)                           
                
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)    
                    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([])
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    ped_local = torch.matmul(robot_local, trans_ped)
                    ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                    ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                       
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]
            
            ped_local_torch = torch.tensor(ped_local_list)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0

            ped_local_slice = ped_local_torch[9:31:3]         
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
          
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]
        
        goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
        relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)
        
        action_IL = []
        metric_waypoint_spacing = 0.25 #normalization        
        for i_traj in range(control_horizon-1):
            traj_relative_mat = to_local_coords_yaw(item["observation.filtered_position"][i_traj + 1, None], unnorm_position[0], current_heading, item["observation.filtered_heading"][i_traj + 1])    
            
            if flip_tf:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, -traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], -traj_relative_mat[1,0]])
            else:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], traj_relative_mat[1,0]])
                            
        action_IL = torch.tensor(action_IL)        
        heading = item["observation.filtered_heading"][:-1]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        
        action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
        action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
        action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing
        
        future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)
        which_dataset = 0

        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")

        if goal_dist == 0:
            goal_is_negative = True
        else:
            goal_is_negative = False
        #[TODO] in GNM dataset, we set goal_dist == 20 when goal_dist is zero (current frame == goal frame) and we give random goal image. But we do in training code. We need to fix this not to confuse the users.
        
        
        action_mask = (
            (goal_dist < self.max_action_distance) and
            (goal_dist > self.min_action_distance) and
            (not goal_is_negative)
        )
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_goal2, dtype=torch.float32),            
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action_IL, dtype=torch.float32),
            torch.as_tensor(goal_dist, dtype=torch.int64),
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(action_mask, dtype=torch.float32),            
            torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)
        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2
        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class ViNTDataset_IL2_10k(Dataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",
        action_format: ActionFormat | str = ActionFormat.WAYPOINT,
        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        context_size: int = 5,
        context_spacing: int = 1,
        dataset_framerate: int = 10,
        image_size: Tuple[int, int] = (120, 160),
        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.context_size = context_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        print("root_init", root)
        """
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
            },
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")        
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }
        """
        self.min_action_distance = 3
        self.max_action_distance = 20

        self.image_transforms = image_transforms        
        self.tolerance_s = 1e-4
        #self.videos_dir = Path(repo_id) / "videos"
        self.videos_dir = Path("/mnt/ephemeral2/frodobots_v2_export/frodobots_dataset_large/videos")
        self.video_backend = "pyav"
        
        img_history_spacing = [i * context_spacing * self.dt for i in range(-context_size, 1)]  # get next image obs too 
        action_future_spacing = [i * action_spacing * self.dt for i in range(action_horizon)]
        self.delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": img_history_spacing,
                "action": action_future_spacing,                
            }

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

        self.total_length = self.dataset_cache["action"].shape[0]
        print("Dataset Cache Loaded", self.dataset_cache["action"].shape)

        # Compute Episode Data Index 
        self.episode_data_index = self.get_episode_data_index(self.dataset_cache["episode_index"])
        print("Episode Index Computed")

    def get_episode_data_index(self, episode_index: list[int]) -> dict[str, torch.Tensor]:
        episode_lengths = []
        current_episode = episode_index[0]
        count = 0

        # Compute Episode Lengths 
        for ep in episode_index:
            if ep == current_episode:
                count += 1
            else:
                episode_lengths.append(count)
                current_episode = ep
                count = 1
        episode_lengths.append(count)  # Append the last episode's length

        # Compute Cumulative Lengths
        cumulative_lengths = list(accumulate(episode_lengths))
        return {
            "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
            "to": torch.LongTensor(cumulative_lengths),
        }
        
    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))
                                            
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist2 = min(8, episode_length_remaining)

        # Add the goal to the list of delta timestamps
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing]  + [goal_dist2 * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        control_horizon = 8+1
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing]
        
        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5    
        image_obs = self._image_transforms(load_from_videos_10k(
            {"observation.images.front": item["observation.images.front"][:-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        image_goal2 = self._image_transforms(load_from_videos_10k(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf) #for inverse dynamics model
                
        image_goal = self._image_transforms(load_from_videos_10k(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
           
        image_current, image_raw = self._image_transforms_depth(load_from_videos_10k(
            {"observation.images.front": item["observation.images.front"][-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf)
        
        
        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)                           
                
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)    
                    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([])
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    ped_local = torch.matmul(robot_local, trans_ped)
                    ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                    ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                       
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]
            
            ped_local_torch = torch.tensor(ped_local_list)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0

            ped_local_slice = ped_local_torch[9:31:3]         
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
          
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]
        
        goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
        relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)
        
        action_IL = []
        metric_waypoint_spacing = 0.25 #normalization        
        for i_traj in range(control_horizon-1):
            traj_relative_mat = to_local_coords_yaw(item["observation.filtered_position"][i_traj + 1, None], unnorm_position[0], current_heading, item["observation.filtered_heading"][i_traj + 1])    
            
            if flip_tf:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, -traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], -traj_relative_mat[1,0]])
            else:
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], traj_relative_mat[1,0]])
                            
        action_IL = torch.tensor(action_IL)        
        heading = item["observation.filtered_heading"][:-1]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        
        action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
        action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
        action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing
        
        future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)
        which_dataset = 0

        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")

        if goal_dist == 0:
            goal_is_negative = True
        else:
            goal_is_negative = False
        #[TODO] in GNM dataset, we set goal_dist == 20 when goal_dist is zero (current frame == goal frame) and we give random goal image. But we do in training code. We need to fix this not to confuse the users.
        
        
        action_mask = (
            (goal_dist < self.max_action_distance) and
            (goal_dist > self.min_action_distance) and
            (not goal_is_negative)
        )
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_goal2, dtype=torch.float32),            
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action_IL, dtype=torch.float32),
            torch.as_tensor(goal_dist, dtype=torch.int64),
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(action_mask, dtype=torch.float32),            
            torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)
        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2
        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class ViNTLeRobotDataset_annotate(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",

        action_format: ActionFormat | str = ActionFormat.WAYPOINT,

        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        context_size: int = 5,
        context_spacing: int = 1,

        dataset_framerate: int = 10,

        image_size: Tuple[int, int] = (120, 160),

        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.context_size = context_size
        #self.ped_past_size = ped_past_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.utm_position": [0.0],
                "observation.compass_heading": [0.0], 
                "observation.gyroscope": [0.0],
                "observation.accelerometer": [0.0],
                "observation.wheel_rpm": [0.0],
                "observation.latitude": [0.0],
                "observation.longitude": [0.0],                
                "action": [0.0],       
                "action_original": [0.0],       
                "observation.utm_zone_letter": [0.0],
                "observation.utm_zone_number": [0.0],    
                "index": [0.0],               
                "observation.images.front": [0.0],                   
                #"observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                #"action": [i * self.dt for i in range(action_spacing * action_horizon)],
                #"action": [i * action_spacing * self.dt for i in range(action_horizon)],                
                #"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)],
            },
            #video_backend = "video_reader",
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        #for k, v in self.dataset_cache.items():
        #    print(k, v.shape) 
        
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

        self.min_action_distance = 0
        self.max_action_distance = 10

    def _image_transforms(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3

        #print(img.size())
        """
        if original_width / original_height > target_aspect:
            target_width = int(original_height * target_aspect)
            crop_left = (original_width - target_width) // 2
            print(crop_left, original_height, target_width)
            img = TF.resized_crop(img, 0, crop_left, original_height, target_width, self.image_size)
        else:
            target_height = int(original_width / target_aspect)
            crop_top = (original_height - target_height) // 2
            #print(target_height, crop_top)
            img = TF.resized_crop(img, crop_top, 0, target_height, original_width, self.image_size)
        """
        img = TF.resize(img, self.image_size)
        return img

    def _image_transforms_depth(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #
        #img = TF.resize(img, (384, 512)) #for Zoedepth   
        
        """
        input_size = (616, 1064)
        h, w = img.shape[-2:]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        """  
        #img = TF.resize(img, (616, 1064)) #for metric3D              
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __getitem__(self, idx):
        #print(self.dataset_cache.keys())
        
        #print("data length", len(self.dataset_cache["episode_index"]))
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        #goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        #goal_dist = int(np.random.exponential(scale=self.goal_horizon)) #noriaki comment out
        #goal_dist = min(goal_dist, episode_length_remaining.item() - 1) #noriaki comment out
        goal_dist = min(self.goal_horizon, episode_length_remaining.item() - 1) 

        # Add the goal to the list of delta timestamps
        #print("self", self.delta_timestamps["observation.images.front"], self.delta_timestamps["action"])
        #delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = self.delta_timestamps
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }
        
        """
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(self.goal_horizon)
        ] + [goal_dist * self.dt * self.action_spacing]

        delta_timestamps["observation.utm_position"] = delta_timestamps["observation.compass_heading"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing]   
        """
        #print("check", delta_timestamps["observation.images.front"])
        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )

        image_current, image_raw = self._image_transforms_depth(load_from_videos(
            {"observation.images.front": item["observation.images.front"][0]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"])


        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]
        
        goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
        relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)
                
        #print("filtered_position", item["observation.filtered_position"])
        #print("filtered_heading", item["observation.filtered_heading"])
        #print("relative_mat", relative_mat)
        
        filtered_position = item["observation.filtered_position"]
        filtered_heading = item["observation.filtered_heading"]
        utm_position = item["observation.utm_position"]
        compass_heading = item["observation.compass_heading"]        
        gyroscope = item["observation.gyroscope"]   
        accelerometer = item["observation.accelerometer"] 
        wheel_rpm = item["observation.wheel_rpm"] 
        action = item["action"] 
        action_original = item["action_original"] 
        episode_index = item["episode_index"] 
        frame_index = item["frame_index"] 
        timestamp = item["timestamp"]
        utm_zone_letter = item["observation.utm_zone_letter"]
        utm_zone_number = item["observation.utm_zone_number"]    
        lat = item["observation.latitude"]
        lon = item["observation.longitude"]           
        index = item["index"]                  
                                          
        video_id_number = item["observation.images.front"][0]["path"].split("_")                                          
        #print(utm_zone_letter, utm_zone_number, index, video_id_number[1], video_id_number[2])                     
                                                                        
        return (
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(filtered_position, dtype=torch.float32),     
            torch.as_tensor(filtered_heading, dtype=torch.float32),         
            torch.as_tensor(utm_position, dtype=torch.float32),        
            torch.as_tensor(compass_heading, dtype=torch.float32),                            
            torch.as_tensor(gyroscope, dtype=torch.float32),      
            torch.as_tensor(accelerometer, dtype=torch.float32),         
            torch.as_tensor(wheel_rpm, dtype=torch.float32),        
            torch.as_tensor(action, dtype=torch.float32),                            
            torch.as_tensor(action_original, dtype=torch.float32),       
            torch.as_tensor(episode_index, dtype=torch.float32),              
            torch.as_tensor(frame_index, dtype=torch.float32),       
            torch.as_tensor(timestamp, dtype=torch.float32),                                                               
            torch.as_tensor(utm_zone_letter, dtype=torch.float32),       
            torch.as_tensor(utm_zone_number, dtype=torch.float32),         
            torch.as_tensor(image_current, dtype=torch.float32),          
            torch.as_tensor(image_raw, dtype=torch.float32),                       
            torch.as_tensor(lat, dtype=torch.float32),       
            torch.as_tensor(lon, dtype=torch.float32),                                    
            torch.as_tensor(float(video_id_number[1]), dtype=torch.float32),       
            torch.as_tensor(float(video_id_number[2]), dtype=torch.float32), 
            torch.as_tensor(relative_mat, dtype=torch.float32)                 
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)

        # heading = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        # initial_heading = heading[indices]
        # final_heading = heading[target_indices]

        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class ViNTDataset_annotate_10k(Dataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",

        action_format: ActionFormat | str = ActionFormat.WAYPOINT,

        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        context_size: int = 5,
        context_spacing: int = 1,

        dataset_framerate: int = 10,

        image_size: Tuple[int, int] = (120, 160),

        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.context_size = context_size
        #self.ped_past_size = ped_past_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        print("root_init", root)
        """
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.utm_position": [0.0],
                "observation.compass_heading": [0.0], 
                "observation.gyroscope": [0.0],
                "observation.accelerometer": [0.0],
                "observation.wheel_rpm": [0.0],
                "observation.latitude": [0.0],
                "observation.longitude": [0.0],                
                "action": [0.0],       
                "action_original": [0.0],       
                "observation.utm_zone_letter": [0.0],
                "observation.utm_zone_number": [0.0],    
                "index": [0.0],               
                "observation.images.front": [0.0],                   
                #"observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                #"action": [i * self.dt for i in range(action_spacing * action_horizon)],
                #"action": [i * action_spacing * self.dt for i in range(action_horizon)],                
                #"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)],
            },
            #video_backend = "video_reader",
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        #for k, v in self.dataset_cache.items():
        #    print(k, v.shape) 
        
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }
        """

        self.min_action_distance = 0
        self.max_action_distance = 10

        self.image_transforms = image_transforms        
        #self.tolerance_s = 1e-4
        self.tolerance_s = 2e-4        
        #self.videos_dir = Path(repo_id) / "videos"
        self.videos_dir = Path("/mnt/ephemeral2/frodobots_v2_export/frodobots_dataset_large/videos")
        self.video_backend = "pyav"
        
        img_history_spacing = [i * context_spacing * self.dt for i in range(-context_size, 1)]  # get next image obs too 
        action_future_spacing = [i * action_spacing * self.dt for i in range(action_horizon)]
        self.delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.utm_position": [0.0],
                "observation.compass_heading": [0.0], 
                "observation.gyroscope": [0.0],
                "observation.accelerometer": [0.0],
                "observation.wheel_rpm": [0.0],
                "observation.latitude": [0.0],
                "observation.longitude": [0.0],                
                "action": [0.0],       
                "action_original": [0.0],       
                "observation.utm_zone_letter": [0.0],
                "observation.utm_zone_number": [0.0],    
                "index": [0.0],               
                "observation.images.front": [0.0],                
            }

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

        self.total_length = self.dataset_cache["action"].shape[0]
        print("Dataset Cache Loaded", self.dataset_cache["action"].shape)

        # Compute Episode Data Index 
        self.episode_data_index = self.get_episode_data_index(self.dataset_cache["episode_index"])
        print("Episode Index Computed")

    def _image_transforms(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3

        #print(img.size())
        """
        if original_width / original_height > target_aspect:
            target_width = int(original_height * target_aspect)
            crop_left = (original_width - target_width) // 2
            print(crop_left, original_height, target_width)
            img = TF.resized_crop(img, 0, crop_left, original_height, target_width, self.image_size)
        else:
            target_height = int(original_width / target_aspect)
            crop_top = (original_height - target_height) // 2
            #print(target_height, crop_top)
            img = TF.resized_crop(img, crop_top, 0, target_height, original_width, self.image_size)
        """
        img = TF.resize(img, self.image_size)
        return img

    def _image_transforms_depth(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #
        #img = TF.resize(img, (384, 512)) #for Zoedepth   
        
        """
        input_size = (616, 1064)
        h, w = img.shape[-2:]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        """  
        #img = TF.resize(img, (616, 1064)) #for metric3D              
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def get_episode_data_index(self, episode_index: list[int]) -> dict[str, torch.Tensor]:
        episode_lengths = []
        current_episode = episode_index[0]
        count = 0

        # Compute Episode Lengths 
        for ep in episode_index:
            if ep == current_episode:
                count += 1
            else:
                episode_lengths.append(count)
                current_episode = ep
                count = 1
        episode_lengths.append(count)  # Append the last episode's length

        # Compute Cumulative Lengths
        cumulative_lengths = list(accumulate(episode_lengths))
        return {
            "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
            "to": torch.LongTensor(cumulative_lengths),
        }

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        #print(self.dataset_cache.keys())
        
        #print("data length", len(self.dataset_cache["episode_index"]))
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        #goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        #goal_dist = int(np.random.exponential(scale=self.goal_horizon)) #noriaki comment out
        #goal_dist = min(goal_dist, episode_length_remaining.item() - 1) #noriaki comment out
        goal_dist = min(self.goal_horizon, episode_length_remaining.item() - 1) 

        # Add the goal to the list of delta timestamps
        #print("self", self.delta_timestamps["observation.images.front"], self.delta_timestamps["action"])
        #delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = self.delta_timestamps
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }
        
        """
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(self.goal_horizon)
        ] + [goal_dist * self.dt * self.action_spacing]

        delta_timestamps["observation.utm_position"] = delta_timestamps["observation.compass_heading"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing]   
        """
        #print("check", delta_timestamps["observation.images.front"])
        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        """
        image_current, image_raw = self._image_transforms_depth(load_from_videos_10k(
            {"observation.images.front": item["observation.images.front"][0]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"])
        """

        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]
        
        goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
        relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)
                
        #print("filtered_position", item["observation.filtered_position"])
        #print("filtered_heading", item["observation.filtered_heading"])
        #print("relative_mat", relative_mat)
        
        filtered_position = item["observation.filtered_position"]
        filtered_heading = item["observation.filtered_heading"]
        utm_position = item["observation.utm_position"]
        compass_heading = item["observation.compass_heading"]        
        gyroscope = item["observation.gyroscope"]   
        accelerometer = item["observation.accelerometer"] 
        wheel_rpm = item["observation.wheel_rpm"] 
        action = item["action"] 
        action_original = item["action_original"] 
        episode_index = item["episode_index"] 
        frame_index = item["frame_index"] 
        timestamp = item["timestamp"]
        utm_zone_letter = item["observation.utm_zone_letter"]
        utm_zone_number = item["observation.utm_zone_number"]    
        lat = item["observation.latitude"]
        lon = item["observation.longitude"]           
        index = item["index"]                  
                                          
        #video_id_number = item["observation.images.front"][0]["path"].split("_")                                          
        #print(utm_zone_letter, utm_zone_number, index, video_id_number[1], video_id_number[2])                     
                                                                        
        return (
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(filtered_position, dtype=torch.float32),     
            torch.as_tensor(filtered_heading, dtype=torch.float32),         
            torch.as_tensor(utm_position, dtype=torch.float32),        
            torch.as_tensor(compass_heading, dtype=torch.float32),                            
            torch.as_tensor(gyroscope, dtype=torch.float32),      
            torch.as_tensor(accelerometer, dtype=torch.float32),         
            torch.as_tensor(wheel_rpm, dtype=torch.float32),        
            torch.as_tensor(action, dtype=torch.float32),                            
            torch.as_tensor(action_original, dtype=torch.float32),       
            torch.as_tensor(episode_index, dtype=torch.float32),              
            torch.as_tensor(frame_index, dtype=torch.float32),       
            torch.as_tensor(timestamp, dtype=torch.float32),                                                               
            torch.as_tensor(utm_zone_letter, dtype=torch.float32),       
            torch.as_tensor(utm_zone_number, dtype=torch.float32),         
            #torch.as_tensor(image_current, dtype=torch.float32),          
            #torch.as_tensor(image_raw, dtype=torch.float32),                       
            torch.as_tensor(lat, dtype=torch.float32),       
            torch.as_tensor(lon, dtype=torch.float32),                                    
            #torch.as_tensor(float(video_id_number[1]), dtype=torch.float32),       
            #torch.as_tensor(float(video_id_number[2]), dtype=torch.float32), 
            torch.as_tensor(relative_mat, dtype=torch.float32)                 
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)

        # heading = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        # initial_heading = heading[indices]
        # final_heading = heading[target_indices]

        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class ViNTLeRobotDataset_IL2_gps_map2_crop_shadow(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        video: str,
        root: Path | None,
        split: str = "train",

        action_format: ActionFormat | str = ActionFormat.WAYPOINT,

        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        goal_horizon2: int = 20,        
        context_size: int = 5,
        context_spacing: int = 1,

        dataset_framerate: int = 10,

        image_size: Tuple[int, int] = (120, 160),

        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.goal_horizon2 = goal_horizon2        
        self.context_size = context_size
        #self.ped_past_size = ped_past_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson

        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.latitude": [0.0],
                "observation.longitude": [0.0],     
                "observation.compass_heading": [0.0],         
                "observation.utm_position": [0.0],
                "observation.utm_zone_letter": [0.0],     
                "observation.utm_zone_number": [0.0],                                           
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                #"action": [i * self.dt for i in range(action_spacing * action_horizon)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
                #"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)],
            },
            #video_backend = "video_reader",
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        #for k, v in self.dataset_cache.items():
        #    print(k, v.shape) 
        
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

        #self.min_action_distance = 0
        #self.max_action_distance = 100
        self.min_action_distance = 3
        self.max_action_distance = 20
        #self.map_image_gen = MapTileCache(map_link)
        self.map_image_gen = MapTileCache(map_link)
        
        self.transform_PIL_tensor = transforms.ToTensor()
        
    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3

        #print(img.size())ped_list_no_trans
        
        """
        if original_width / original_height > target_aspect:
            target_width = int(original_height * target_aspect)
            crop_left = (original_width - target_width) // 2
            print(crop_left, original_height, target_width)
            img = TF.resized_crop(img, 0, crop_left, original_height, target_width, self.image_size)
        else:
            target_height = int(original_width / target_aspect)
            crop_top = (original_height - target_height) // 2
            #print(target_height, crop_top)
            img = TF.resized_crop(img, crop_top, 0, target_height, original_width, self.image_size)
        """
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img

    def _image_transforms_rand_crop(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        
        #print(img.size())
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3

        #print(img.size())
        
        """
        if original_width / original_height > target_aspect:
            target_width = int(original_height * target_aspect)
            crop_left = (original_width - target_width) // 2
            print(crop_left, original_height, target_width)
            img = TF.resized_crop(img, 0, crop_left, original_height, target_width, self.image_size)
        else:
            target_height = int(original_width / target_aspect)
            crop_top = (original_height - target_height) // 2
            #print(target_height, crop_top)
            img = TF.resized_crop(img, crop_top, 0, target_height, original_width, self.image_size)
        """
        top = np.random.randint(0, 20)
        height = 144 - np.random.randint(0, 20) - top 
        left = np.random.randint(0, 25) 
        width = 256 - 2*left
        
        img = TF.crop(img, int(top), int(left), int(height), int(width))
        #print(img.size(), top, left, height, width)
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img     

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #
        #img = TF.resize(img, (384, 512)) #for Zoedepth   
        
        """
        input_size = (616, 1064)
        h, w = img.shape[-2:]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        """  
        #img = TF.resize(img, (616, 1064)) #for metric3D            
        
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))
                      
        #print(img.size(), img_rsize.size())
                      
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def _latlon_to_utm(self, lat, lon):
        """ Convert latitude and longitude to UTM coordinates. """
        easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
        return easting, northing, zone_number, zone_letter

    def _utm_to_latlon(self, easting, northing, zone_number, zone_letter):
        """ Convert UTM coordinates back to latitude and longitude. """
        lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)
        return lat, lon

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
        new_heading = (heading - theta) 
    
        # Corrected transformation: X moves forward, Y moves left
        delta_easting = np.sqrt(X**2 + Y**2) * np.sin(new_heading) 
        delta_northing = np.sqrt(X**2 + Y**2) * np.cos(new_heading) 

        # New position in UTM coordinates
        new_easting = easting + delta_easting
        new_northing = northing + delta_northing

        # Convert back to latitude and longitude
        new_lat, new_lon = self._utm_to_latlon(new_easting, new_northing, zone_number, zone_letter)

        # Update heading (subtract for CCW rotation)
        new_heading = (heading - theta)  

        return new_lat, new_lon, new_heading

    def _add_shadow_to_tensor_image(self, img_tensor: torch.Tensor, num_shadows=1, shadow_intensity=0.6):
        if img_tensor.ndim != 3 or img_tensor.shape[0] != 3:
            shadow_img_list = []
            for i in range(img_tensor.shape[0]):
                C, H, W = img_tensor[i].shape
                shadow_mask = torch.zeros((H, W), dtype=torch.float32)

                for _ in range(num_shadows):
                    num_points = random.randint(3, 8)
                    points = [(random.randint(0, W-1), random.randint(int(0.0*H), H-1)) for _ in range(num_points)]
        
                    # Create a binary mask for the polygon using torchvision (or PIL + numpy fallback)
                    try:
                        from torchvision.utils import draw_segmentation_masks
                        import torchvision.transforms.functional as F
                        from torchvision.ops import masks_to_boxes

                        # Generate mask using torchvision.draw_segmentation_masks if available
                        import numpy as np
                        import cv2

                        mask_np = np.zeros((H, W), dtype=np.uint8)
                        cv2.fillPoly(mask_np, [np.array(points, dtype=np.int32)], color=1)
                        shadow_mask += torch.from_numpy(mask_np).float()

                    except ImportError:
                       raise ImportError("This function requires torchvision and OpenCV (cv2) to be installed.")
                       
                shadow_mask = torch.clamp(shadow_mask, 0, 1)

                # Randomize shadow darkness
                shadow_intensity = shadow_intensity

                # Blend shadow into image
                shadow_factor = 1 - (shadow_mask * shadow_intensity)  # [H, W]
                shadow_factor = shadow_factor.unsqueeze(0).expand_as(img_tensor[i])  # [C, H, W]
    
                #print(img_tensor.max(), img_tensor.min())    
                #print(shadow_factor.max(), shadow_factor.min())
                shadowed_img_c = img_tensor[i] * shadow_factor
                shadow_img_list.append(shadowed_img_c.unsqueeze(0))
            shadowed_img = torch.cat(shadow_img_list, axis=0)
                                   
        else:
            C, H, W = img_tensor.shape
            shadow_mask = torch.zeros((H, W), dtype=torch.float32)

            for _ in range(num_shadows):
                num_points = random.randint(3, 8)
                points = [(random.randint(0, W-1), random.randint(int(0.0*H), H-1)) for _ in range(num_points)]
        
                # Create a binary mask for the polygon using torchvision (or PIL + numpy fallback)
                try:
                    from torchvision.utils import draw_segmentation_masks
                    import torchvision.transforms.functional as F
                    from torchvision.ops import masks_to_boxes

                    # Generate mask using torchvision.draw_segmentation_masks if available
                    import numpy as np
                    import cv2

                    mask_np = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(mask_np, [np.array(points, dtype=np.int32)], color=1)
                    shadow_mask += torch.from_numpy(mask_np).float()

                except ImportError:
                    raise ImportError("This function requires torchvision and OpenCV (cv2) to be installed.")

            shadow_mask = torch.clamp(shadow_mask, 0, 1)

            # Randomize shadow darkness
            shadow_intensity = shadow_intensity

            # Blend shadow into image
            shadow_factor = 1 - (shadow_mask * shadow_intensity)  # [H, W]
            shadow_factor = shadow_factor.unsqueeze(0).expand_as(img_tensor)  # [C, H, W]
    
            #print(img_tensor.max(), img_tensor.min())    
            #print(shadow_factor.max(), shadow_factor.min())
            shadowed_img = img_tensor * shadow_factor
        return shadowed_img

    def __getitem__(self, idx):
        
        #print("data length", len(self.dataset_cache["episode_index"]))
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist2 = min(8, episode_length_remaining)
        goal_dist_gps = np.random.randint(0, min(self.goal_horizon2, episode_length_remaining))    
        #goal_dist_gps = min(self.goal_horizon2, episode_length_remaining)   
        #goal_dist = int(np.random.exponential(scale=self.goal_horizon)) #noriaki comment out
        #goal_dist = min(goal_dist, episode_length_remaining.item() - 1) #noriaki comment out

        # Add the goal to the list of delta timestamps
        #print("self", self.delta_timestamps["observation.images.front"], self.delta_timestamps["action"])
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] + [goal_dist2 * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        control_horizon = 8+1
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing]
        #delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
        #    i * self.dt * self.action_spacing
        #    for i in range(self.goal_horizon)
        #] + [goal_dist * self.dt * self.action_spacing]
        
        #print("check", delta_timestamps["observation.images.front"])ped_list_no_trans
        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5
        
        num_shadows = random.randint(0, 3)
        shadow_intensity = random.uniform(0, 0.8)           
        #print("observation.images.front", item["observation.images.front"][:])
        #print("observation.filtered_heading", item["observation.filtered_heading"][:])       
        
        image_obs_raw = load_from_videos(
            {"observation.images.front": item["observation.images.front"][:-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"]
        
        #print(image_obs_raw.size())
        #image_obs_raw = self._add_shadow_to_tensor_image(image_obs_raw, num_shadows=num_shadows, shadow_intensity=shadow_intensity)
        ##print(image_obs_raw.size())         
        #image_obs = self._image_transforms(image_obs_raw, flip_tf)
        #image_obs_crop = self._image_transforms_rand_crop(image_obs_raw, flip_tf)
        
        image_obs_raw_s = self._add_shadow_to_tensor_image(image_obs_raw, num_shadows=num_shadows, shadow_intensity=shadow_intensity)
        image_obs_crop = self._image_transforms_rand_crop(image_obs_raw_s, flip_tf)        
        #print(image_obs_raw.size())         
        image_obs = self._image_transforms(image_obs_raw, flip_tf)        
        """
        test = load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )
        """        
        image_goal2 = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf) #for inverse dynamics model
                
        image_goal = self._image_transforms_rand_crop(self._add_shadow_to_tensor_image(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], num_shadows=num_shadows, shadow_intensity=shadow_intensity), flip_tf)
        
        
        image_current, image_raw = self._image_transforms_depth(self._add_shadow_to_tensor_image(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-4]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], num_shadows=num_shadows, shadow_intensity=shadow_intensity), flip_tf)
        
        #self.map_image_gen
        
        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            #print("checking", item_pickle[30], delta_timestamps_sacson["pedestrian.filtered_position"][30], idx)
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)               
                #data = cloudpickle.load(file)               
                
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            #for id_ped in range(-30, 5):
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)    
                    #data = cloudpickle.load(file)    
                    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([])
            #print(ped_select, len(ped_list), ped_list)    
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    #print(trans_cur_inv, trans_fp, trans_ped)
                    ped_local = torch.matmul(robot_local, trans_ped)
                    ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                    ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                       
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)

            #print("value_list", value_list)
            #print("before", ped_local_list)  
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    #print(jj - value_list[ii], jj, value_list[ii]+jj)
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]
                    #ped_local_list[value_list[ii]+jj] = [ped_local_list[value_list[ii]][0] + (jj+1)*delta_p[0], ped_local_list[value_list[ii]][1] + (jj+1)*delta_p[1]]
            
            ped_local_torch = torch.tensor(ped_local_list)
            #ped_local_torch = ped_local_list
            #filtering
            #print("Before", ped_local_torch)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0
            #print("After", ped_local_torch) 
            
            #ped_local_slice = ped_local_list[30:7:-3][::-1] #3fps
            #print(ped_local_torch.size())
            #ped_local_torch = torch.flip(ped_local_torch, dims=[0])[3:30:3]
            ped_local_slice = ped_local_torch[9:31:3]
            #print(ped_local_slice.size(), ped_local_slice, ped_local_torch[30])
            #ped_local_slice = ped_local_torch[30:7:-3][::-1] #3fps            
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            #print(len(ped_local_slice), ped_local_list[30], ped_local_slice)
            #print(ped_local_list)
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
        """
        ped_list_no_trans = [0.0] #dummy
        ped_local_slice = [0.0] #dummy
        ped_local_slice_raw = [0.0] #dummy
        robot_local_slice = [0.0] #dummy
        
        """             
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]

        current_lat = item["observation.latitude"][0]
        current_lon = item["observation.longitude"][0]
        #current_compass = item["observation.compass_heading"][0]  
        current_compass = item["observation.filtered_heading"][0]       
        current_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), flip_tf)
        """
        image_PIL = self.map_image_gen.get_map_view(current_lat, current_lon, current_compass).resize((96, 96), resample=Image.Resampling.LANCZOS)
        image_noflip = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), False)
        image_flip = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), True)
        
        image_noflip_np = 255.0*image_noflip.permute(1, 2, 0).numpy()
        image_flip_np = 255.0*image_flip.permute(1, 2, 0).numpy()        
        #print(image_noflip.size(), image_flip.size())

        img_concat = np.clip(np.concatenate((image_PIL, image_noflip_np, image_flip_np), axis=1), 0, 255).astype(np.uint8)
        print(img_concat.shape)
        
        plt.imshow(img_concat)
        plt.axis('off')  # Hide the axis
        plt.savefig('saved_image.png')
        """
        #goal_lat = item["observation.latitude"][-1]
        #goal_lon = item["observation.longitude"][-1]
        #goal_compass = item["observation.compass_heading"][-1]      
        #goal_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(goal_lat, goal_lon, goal_compass)), flip_tf)
        #print("compass", current_compass, goal_compass)    
        
                                        
        #print(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)
        if flip_tf:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]        
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)
            
            x_loc = relative_mat[0,2]
            y_loc = relative_mat[1,2]
            yaw_loc = np.arctan2(relative_mat[1,0], relative_mat[1,1])
            #print(yaw_loc, goal_heading-current_heading)            
            #filt_goal = item["observation.filtered_position"][-1, None]
            #utm_goal = item["observation.utm_position"][-1, None]
            #let_goal = item["observation.utm_zone_letter"][-1, None]
            #num_goal = item["observation.utm_zone_number"][-1, None]
            #print("dataset", filt_goal, utm_goal, let_goal, num_goal)
            
            new_lat, new_lon, new_heading = self._transform_position(current_lat.item(), current_lon.item(), current_compass.item(), x_loc.item(), y_loc.item(), yaw_loc.item())              
            try:
                goal_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(new_lat, new_lon, new_heading)), flip_tf)
            except Exception as e:
                goal_lat = item["observation.latitude"][-1]
                goal_lon = item["observation.longitude"][-1]
                #goal_compass = item["observation.compass_heading"][-1] 
                goal_compass = item["observation.filtered_heading"][-1]                      
                goal_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(goal_lat, goal_lon, goal_compass)), flip_tf)      
            
            
            #print("before", goal_pos_relative, relative_mat)            
            goal_pos_relative[1] *= -1                
            relative_mat[0,1] *= -1
            relative_mat[1,0] *= -1                                    
            #print("after", goal_pos_relative, goal_dist_gps)
        else:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)        

            x_loc = relative_mat[0,2]
            y_loc = relative_mat[1,2]
            yaw_loc = np.arctan2(relative_mat[1,0], relative_mat[1,1])
            #print(yaw_loc, goal_heading-current_heading)
            
            new_lat, new_lon, new_heading = self._transform_position(current_lat.item(), current_lon.item(), current_compass.item(), x_loc.item(), y_loc.item(), yaw_loc.item())        
            try:
                goal_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(new_lat, new_lon, new_heading)), flip_tf)
            except Exception as e:
                goal_lat = item["observation.latitude"][-1]
                goal_lon = item["observation.longitude"][-1]
                #goal_compass = item["observation.compass_heading"][-1] 
                goal_compass = item["observation.filtered_heading"][-1]                      
                goal_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(goal_lat, goal_lon, goal_compass)), flip_tf)     
                  
         
        action_IL = []
        metric_waypoint_spacing = 0.25 #normalization        
        goal_pos_relative = goal_pos_relative/metric_waypoint_spacing #normalization        
        for i_traj in range(control_horizon-1):
            traj_relative_mat = to_local_coords_yaw(item["observation.filtered_position"][i_traj + 1, None], unnorm_position[0], current_heading, item["observation.filtered_heading"][i_traj + 1])    
            
            if flip_tf:
                #print("FLIP")
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, -traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], -traj_relative_mat[1,0]])
            else:
                #print("NO FLIP")
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], traj_relative_mat[1,0]])
                            
        action_IL = torch.tensor(action_IL)        
        #print(action_IL.size())
                
        # action = einops.reduce(item["action"][:-1], "(a s) d -> a d", reduction="mean", s=self.action_spacing)
        #action_is_pad = einops.reduce(item["action_is_pad"][:-1], "(a s) -> a", reduction="max", s=self.action_spacing)
        heading = item["observation.filtered_heading"][:-1]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        
        #print("action", heading, np.unwrap(heading), np.diff(np.unwrap(heading)))
        action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
        action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
        action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing
        
        #action = [0.0]
        
        future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)
        which_dataset = 0

        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")
        image_flattened_crop = einops.rearrange(image_obs_crop, "... t c h w -> ... (t c) h w")
        #image_flattened = [0.0]
        #image_goal = [0.0]
        #image_current = [0.0]
        #goal_dist = [0.0]
        #goal_pos_relative =                obs_image, 

        #relative_mat = [0.0]        
        #goal_heading = 0.0
        #current_heading = 0.0
        #which_dataset = [0.0]
        #future_positions_unfiltered = [0.0]
        #idx = [0.0]
        #image_raw = [0.0]

        if goal_dist == 0:
            goal_is_negative = True
        else:
            goal_is_negative = False
        #[TODO] in GNM dataset, we set goal_dist == 20 when goal_dist is zero (current frame == goal frame) and we give random goal image. But we do in training code. We need to fix this not to confuse the users.
        
        
        action_mask = (
            (goal_dist < self.max_action_distance) and
            (goal_dist > self.min_action_distance) and
            (not goal_is_negative)
        )
        #print(action_mask, goal_is_negative, goal_dist, self.max_action_distance, self.min_action_distance)
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_flattened_crop, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_goal2, dtype=torch.float32),            
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action_IL, dtype=torch.float32),
            #torch.as_tensor(goal_dist, dtype=torch.int64),
            torch.as_tensor(goal_dist_gps, dtype=torch.int64),            
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            #torch.as_tensor(~action_is_pad, dtype=torch.float32),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(action_mask, dtype=torch.float32),           
            #torch.as_tensor(image_raw, dtype=torch.float32),     
            #torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            #torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(current_map_image, dtype=torch.float32),         
            torch.as_tensor(goal_map_image, dtype=torch.float32),                 
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)

        # heading = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        # initial_heading = heading[indices]
        # final_heading = heading[target_indices]

        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class ViNTLeRobotDataset_IL2_gps_map2_crop_shadow_MMN(LeRobotDataset):
    def __init__(
        self,
        action_tokenizer: PreTrainedTokenizerBase,
        base_tokenizer: ActionTokenizer,   
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],            
        repo_id: str,
        video: str,
        root: Path | None,
        predict_stop_token: bool = True,         
        split: str = "train",

        action_format: ActionFormat | str = ActionFormat.WAYPOINT,

        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        goal_horizon2: int = 20,        
        context_size: int = 5,
        context_spacing: int = 1,

        dataset_framerate: int = 10,

        image_size: Tuple[int, int] = (120, 160),

        image_transforms: Callable | None = None,
        sacson: bool = False
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.goal_horizon2 = goal_horizon2        
        self.context_size = context_size
        #self.ped_past_size = ped_past_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.sacson = sacson
        
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.prompt_builder = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.image_transform = image_transform        

        print("root_init", root)
        super().__init__(
            repo_id=repo_id,
            video=video,
            root=root,
            split=split,
            image_transforms=image_transforms,
            delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.latitude": [0.0],
                "observation.longitude": [0.0],     
                "observation.compass_heading": [0.0],         
                "observation.utm_position": [0.0],
                "observation.utm_zone_letter": [0.0],     
                "observation.utm_zone_number": [0.0],                                           
                "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
                #"action": [i * self.dt for i in range(action_spacing * action_horizon)],
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],                
                #"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)],
            },
            #video_backend = "video_reader",
        )

        # Build a cache of episode data indices
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        #for k, v in self.dataset_cache.items():
        #    print(k, v.shape) 
        
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }

        #self.min_action_distance = 0
        #self.max_action_distance = 100
        self.min_action_distance = 3
        self.max_action_distance = 20
        #self.map_image_gen = MapTileCache(map_link)
        #self.map_image_gen = MapTileCache(map_link)
        
        self.transform_PIL_tensor = transforms.ToTensor()
        
    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3

        #print(img.size())ped_list_no_trans
        
        """
        if original_width / original_height > target_aspect:
            target_width = int(original_height * target_aspect)
            crop_left = (original_width - target_width) // 2
            print(crop_left, original_height, target_width)
            img = TF.resized_crop(img, 0, crop_left, original_height, target_width, self.image_size)
        else:
            target_height = int(original_width / target_aspect)
            crop_top = (original_height - target_height) // 2
            #print(target_height, crop_top)
            img = TF.resized_crop(img, crop_top, 0, target_height, original_width, self.image_size)
        """
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img

    def _image_transforms_rand_crop(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        
        #print(img.size())
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3

        #print(img.size())
        
        """
        if original_width / original_height > target_aspect:
            target_width = int(original_height * target_aspect)
            crop_left = (original_width - target_width) // 2
            print(crop_left, original_height, target_width)
            img = TF.resized_crop(img, 0, crop_left, original_height, target_width, self.image_size)
        else:
            target_height = int(original_width / target_aspect)
            crop_top = (original_height - target_height) // 2
            #print(target_height, crop_top)
            img = TF.resized_crop(img, crop_top, 0, target_height, original_width, self.image_size)
        """
        top = np.random.randint(0, 20)
        height = 144 - np.random.randint(0, 20) - top 
        left = np.random.randint(0, 25) 
        width = 256 - 2*left
        
        img = TF.crop(img, int(top), int(left), int(height), int(width))
        #print(img.size(), top, left, height, width)
        img = TF.resize(img, self.image_size)
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img     

    def _image_rand_crop_224(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        original_height, original_width = img.shape[-2:]

        top = np.random.randint(0, 20)
        height = 144 - np.random.randint(0, 20) - top 
        left = np.random.randint(0, 25) 
        width = 256 - 2*left
        
        img = TF.crop(img, int(top), int(left), int(height), int(width))
        #print(img.size(), top, left, height, width)
        img = TF.resize(img, (224, 224))
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img     

    def _image_224(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        img = TF.resize(img, (224, 224))
        
        if flip:
            img = torch.flip(img, dims=(-1,))
        
        return img     
        
    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #
        #img = TF.resize(img, (384, 512)) #for Zoedepth   
        
        """
        input_size = (616, 1064)
        h, w = img.shape[-2:]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        """  
        #img = TF.resize(img, (616, 1064)) #for metric3D            
        
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))
                      
        #print(img.size(), img_rsize.size())
                      
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = torch.zeros_like(actions)
            heading = torch.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def _latlon_to_utm(self, lat, lon):
        """ Convert latitude and longitude to UTM coordinates. """
        easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
        return easting, northing, zone_number, zone_letter

    def _utm_to_latlon(self, easting, northing, zone_number, zone_letter):
        """ Convert UTM coordinates back to latitude and longitude. """
        lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)
        return lat, lon

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
        new_heading = (heading - theta) 
    
        # Corrected transformation: X moves forward, Y moves left
        delta_easting = np.sqrt(X**2 + Y**2) * np.sin(new_heading) 
        delta_northing = np.sqrt(X**2 + Y**2) * np.cos(new_heading) 

        # New position in UTM coordinates
        new_easting = easting + delta_easting
        new_northing = northing + delta_northing

        # Convert back to latitude and longitude
        new_lat, new_lon = self._utm_to_latlon(new_easting, new_northing, zone_number, zone_letter)

        # Update heading (subtract for CCW rotation)
        new_heading = (heading - theta)  

        return new_lat, new_lon, new_heading

    def _add_shadow_to_tensor_image(self, img_tensor: torch.Tensor, num_shadows=1, shadow_intensity=0.6):
        if img_tensor.ndim != 3 or img_tensor.shape[0] != 3:
            shadow_img_list = []
            for i in range(img_tensor.shape[0]):
                C, H, W = img_tensor[i].shape
                shadow_mask = torch.zeros((H, W), dtype=torch.float32)

                for _ in range(num_shadows):
                    num_points = random.randint(3, 8)
                    points = [(random.randint(0, W-1), random.randint(int(0.0*H), H-1)) for _ in range(num_points)]
        
                    # Create a binary mask for the polygon using torchvision (or PIL + numpy fallback)
                    try:
                        from torchvision.utils import draw_segmentation_masks
                        import torchvision.transforms.functional as F
                        from torchvision.ops import masks_to_boxes

                        # Generate mask using torchvision.draw_segmentation_masks if available
                        import numpy as np
                        import cv2

                        mask_np = np.zeros((H, W), dtype=np.uint8)
                        cv2.fillPoly(mask_np, [np.array(points, dtype=np.int32)], color=1)
                        shadow_mask += torch.from_numpy(mask_np).float()

                    except ImportError:
                       raise ImportError("This function requires torchvision and OpenCV (cv2) to be installed.")
                       
                shadow_mask = torch.clamp(shadow_mask, 0, 1)

                # Randomize shadow darkness
                shadow_intensity = shadow_intensity

                # Blend shadow into image
                shadow_factor = 1 - (shadow_mask * shadow_intensity)  # [H, W]
                shadow_factor = shadow_factor.unsqueeze(0).expand_as(img_tensor[i])  # [C, H, W]
    
                #print(img_tensor.max(), img_tensor.min())    
                #print(shadow_factor.max(), shadow_factor.min())
                shadowed_img_c = img_tensor[i] * shadow_factor
                shadow_img_list.append(shadowed_img_c.unsqueeze(0))
            shadowed_img = torch.cat(shadow_img_list, axis=0)
                                   
        else:
            C, H, W = img_tensor.shape
            shadow_mask = torch.zeros((H, W), dtype=torch.float32)

            for _ in range(num_shadows):
                num_points = random.randint(3, 8)
                points = [(random.randint(0, W-1), random.randint(int(0.0*H), H-1)) for _ in range(num_points)]
        
                # Create a binary mask for the polygon using torchvision (or PIL + numpy fallback)
                try:
                    from torchvision.utils import draw_segmentation_masks
                    import torchvision.transforms.functional as F
                    from torchvision.ops import masks_to_boxes

                    # Generate mask using torchvision.draw_segmentation_masks if available
                    import numpy as np
                    import cv2

                    mask_np = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(mask_np, [np.array(points, dtype=np.int32)], color=1)
                    shadow_mask += torch.from_numpy(mask_np).float()

                except ImportError:
                    raise ImportError("This function requires torchvision and OpenCV (cv2) to be installed.")

            shadow_mask = torch.clamp(shadow_mask, 0, 1)

            # Randomize shadow darkness
            shadow_intensity = shadow_intensity

            # Blend shadow into image
            shadow_factor = 1 - (shadow_mask * shadow_intensity)  # [H, W]
            shadow_factor = shadow_factor.unsqueeze(0).expand_as(img_tensor)  # [C, H, W]
    
            #print(img_tensor.max(), img_tensor.min())    
            #print(shadow_factor.max(), shadow_factor.min())
            shadowed_img = img_tensor * shadow_factor
        return shadowed_img

    def __getitem__(self, idx):
        
        #print("data length", len(self.dataset_cache["episode_index"]))
        # Sample a goal timestamp        
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist2 = min(8, episode_length_remaining)
        goal_dist_gps = np.random.randint(0, min(self.goal_horizon2, episode_length_remaining))    
        #goal_dist_gps = min(self.goal_horizon2, episode_length_remaining)   
        #goal_dist = int(np.random.exponential(scale=self.goal_horizon)) #noriaki comment out
        #goal_dist = min(goal_dist, episode_length_remaining.item() - 1) #noriaki comment out

        # Add the goal to the list of delta timestamps
        #print("self", self.delta_timestamps["observation.images.front"], self.delta_timestamps["action"])
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}
        delta_timestamps = {
            k: list(v) + [goal_dist * self.dt * self.action_spacing] + [goal_dist2 * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing] if v is not None else None for k, v in delta_timestamps.items()
        }

        control_horizon = 8+1
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing
            for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing]
        #delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
        #    i * self.dt * self.action_spacing
        #    for i in range(self.goal_horizon)
        #] + [goal_dist * self.dt * self.action_spacing]
        
        #print("check", delta_timestamps["observation.images.front"])ped_list_no_trans
        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )
        
        flip_tf = random.random() > 0.5
        
        num_shadows = random.randint(0, 3)
        shadow_intensity = random.uniform(0, 0.8)  
        shadow_intensity = 0.0         
        #print("observation.images.front", item["observation.images.front"][:])
        #print("observation.filtered_heading", item["observation.filtered_heading"][:])       
        
        image_obs_raw = load_from_videos(
            {"observation.images.front": item["observation.images.front"][:-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"]
        
        #print(image_obs_raw.size())
        #image_obs_raw = self._add_shadow_to_tensor_image(image_obs_raw, num_shadows=num_shadows, shadow_intensity=shadow_intensity)
        ##print(image_obs_raw.size())         
        #image_obs = self._image_transforms(image_obs_raw, flip_tf)
        #image_obs_crop = self._image_transforms_rand_crop(image_obs_raw, flip_tf)
        
        image_obs_raw_s = self._add_shadow_to_tensor_image(image_obs_raw, num_shadows=num_shadows, shadow_intensity=shadow_intensity)
        image_obs_crop = self._image_transforms_rand_crop(image_obs_raw_s, flip_tf)        
        #print(image_obs_raw.size())         
        image_obs = self._image_transforms(image_obs_raw, flip_tf)
            
        """
        test = load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )
        """        
        image_goal2 = self._image_transforms(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-2]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], flip_tf) #for inverse dynamics model
                
        image_goal = self._image_transforms_rand_crop(self._add_shadow_to_tensor_image(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], num_shadows=num_shadows, shadow_intensity=shadow_intensity), flip_tf)

        image_goal_topil = self._image_rand_crop_224(self._add_shadow_to_tensor_image(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-1]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], num_shadows=num_shadows, shadow_intensity=shadow_intensity), flip_tf)        
        
        image_current_shadow = self._add_shadow_to_tensor_image(load_from_videos(
            {"observation.images.front": item["observation.images.front"][-4]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"], num_shadows=num_shadows, shadow_intensity=shadow_intensity)
        
        image_current, image_raw = self._image_transforms_depth(image_current_shadow, flip_tf)

        image_obs_topil = self._image_rand_crop_224(image_current_shadow, flip_tf)       
        #image_obs_topil = self._image_rand_crop_224(self._add_shadow_to_tensor_image(image_obs_raw[-1], num_shadows=num_shadows, shadow_intensity=shadow_intensity), flip_tf)         
        #self.map_image_gen
        
        if self.sacson:
            delta_timestamps_sacson = {"pedestrian.filtered_position": [i * self.dt for i in range(-30, 5)]}
            item_pickle = load_pickle(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps_sacson,
            )      
            #print("checking", item_pickle[30], delta_timestamps_sacson["pedestrian.filtered_position"][30], idx)
            episode_length_prev = idx - self.episode_data_index["from"][ep_id]
            with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(idx)) + ".pkl", 'rb') as file:
                data = pickle.load(file)               
                #data = cloudpickle.load(file)               
                
            ped_dist = 1000000000.0
            ped_select = None
            for ip in data.keys():     
                dist = data[ip][0]**2 + data[ip][1]**2
                if dist < ped_dist:
                    ped_dist = dist
                    ped_select = ip
            
            ped_list = []            
            #for id_ped in range(-30, 5):
            for id_ped in item_pickle:            
                with open("/media/noriaki/Noriaki_Data/frodobots_dataset/ped_est/" + str(int(id_ped)) + ".pkl", 'rb') as file:
                    data = pickle.load(file)    
                    #data = cloudpickle.load(file)    
                    
                if ped_select in data.keys():                
                    ped_list.append(data[ped_select][0:2])   
                else:
                    ped_list.append([])
            #print(ped_select, len(ped_list), ped_list)    
            
            head_trans = torch.from_numpy(self.dataset_cache["observation.filtered_heading"][item_pickle])            
            pos_trans = torch.from_numpy(self.dataset_cache["observation.filtered_position"][item_pickle]) 
            
            trans_cur_inv = torch.linalg.inv(trans_mat(pos_trans[30], head_trans[30]))
            
            ped_list_notrans = []
            ped_local_list = []
            robot_local_list = []            
            for istep in range(len(ped_list)):
                trans_fp = trans_mat(pos_trans[istep], head_trans[istep])            
                robot_local = torch.matmul(trans_cur_inv, trans_fp)
                robot_local_list.append([robot_local[0,2], robot_local[1,2]]) 
                if ped_list[istep] != []:
                    trans_ped = torch.from_numpy(trans_mat([ped_list[istep][1], -ped_list[istep][0]], 0.0))
                    #print(trans_cur_inv, trans_fp, trans_ped)
                    ped_local = torch.matmul(robot_local, trans_ped)
                    ped_local_list.append([ped_local[0,2], ped_local[1,2]])
                    ped_list_notrans.append([ped_list[istep][1], -ped_list[istep][0]])                                       
                else:
                    ped_local_list.append([0.0, 0.0])
                    ped_list_notrans.append([0.0, 0.0])
            
            ped_local_list_raw = ped_local_list.copy()
            value_list = []
            for istep in range(len(ped_list)):
                if ped_local_list[istep] != [0.0, 0.0]:
                     value_list.append(istep)

            #print("value_list", value_list)
            #print("before", ped_local_list)  
            
            #interpolation                    
            for ii in range(len(value_list)-1):
                gap = float(value_list[ii+1] - value_list[ii])              
                delta_p = [(ped_local_list[value_list[ii+1]][0] - ped_local_list[value_list[ii]][0])/gap, (ped_local_list[value_list[ii+1]][1] - ped_local_list[value_list[ii]][1])/gap]
                for jj in range(value_list[ii]+1, value_list[ii+1], 1):
                    #print(jj - value_list[ii], jj, value_list[ii]+jj)
                    ped_local_list[jj] = [ped_local_list[value_list[ii]][0] + (jj - value_list[ii])*delta_p[0], ped_local_list[value_list[ii]][1] + (jj - value_list[ii])*delta_p[1]]
                    #ped_local_list[value_list[ii]+jj] = [ped_local_list[value_list[ii]][0] + (jj+1)*delta_p[0], ped_local_list[value_list[ii]][1] + (jj+1)*delta_p[1]]
            
            ped_local_torch = torch.tensor(ped_local_list)
            #ped_local_torch = ped_local_list
            #filtering
            #print("Before", ped_local_torch)
            for jp in range(5):
                for ip in range(len(ped_list)-2):   
                    if ped_local_torch[ip][0] != 0.0 and ped_local_torch[ip][1] != 0.0 and ped_local_torch[ip+1][0] != 0.0 and ped_local_torch[ip+1][1] != 0.0 and ped_local_torch[ip+2][0] != 0.0 and ped_local_torch[ip+2][1] != 0.0:
                        ped_local_torch[ip+1] = (ped_local_torch[ip] + ped_local_torch[ip+1] + ped_local_torch[ip+2])/3.0
            #print("After", ped_local_torch) 
            
            #ped_local_slice = ped_local_list[30:7:-3][::-1] #3fps
            #print(ped_local_torch.size())
            #ped_local_torch = torch.flip(ped_local_torch, dims=[0])[3:30:3]
            ped_local_slice = ped_local_torch[9:31:3]
            #print(ped_local_slice.size(), ped_local_slice, ped_local_torch[30])
            #ped_local_slice = ped_local_torch[30:7:-3][::-1] #3fps            
            ped_local_slice_raw = ped_local_list_raw[30:7:-3][::-1] #3fps  
            robot_local_slice = robot_local_list[30:7:-3][::-1] 
            #print(len(ped_local_slice), ped_local_list[30], ped_local_slice)
            #print(ped_local_list)
            ped_list_no_trans = ped_list_notrans[30:7:-3][::-1]
                               
        else:
            ped_list_no_trans = [0.0] #dummy
            ped_local_slice = [0.0] #dummy
            ped_local_slice_raw = [0.0] #dummy
            robot_local_slice = [0.0] #dummy
        """
        ped_list_no_trans = [0.0] #dummy
        ped_local_slice = [0.0] #dummy
        ped_local_slice_raw = [0.0] #dummy
        robot_local_slice = [0.0] #dummy
        
        """             
        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]

        current_lat = item["observation.latitude"][0]
        current_lon = item["observation.longitude"][0]
        #current_compass = item["observation.compass_heading"][0]  
        current_compass = item["observation.filtered_heading"][0]      
        
        #cur_map_PIL = self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)) 
        #current_map_image = self._image_transforms(cur_map_PIL, flip_tf)
        #current_map_PILimage = to_pil_image(self._image_224(cur_map_PIL, flip_tf))
        """
        image_PIL = self.map_image_gen.get_map_view(current_lat, current_lon, current_compass).resize((96, 96), resample=Image.Resampling.LANCZOS)
        image_noflip = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), False)
        image_flip = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(current_lat, current_lon, current_compass)), True)
        
        image_noflip_np = 255.0*image_noflip.permute(1, 2, 0).numpy()
        image_flip_np = 255.0*image_flip.permute(1, 2, 0).numpy()        
        #print(image_noflip.size(), image_flip.size())

        img_concat = np.clip(np.concatenate((image_PIL, image_noflip_np, image_flip_np), axis=1), 0, 255).astype(np.uint8)
        print(img_concat.shape)
        
        plt.imshow(img_concat)
        plt.axis('off')  # Hide the axis
        plt.savefig('saved_image.png')
        """
        #goal_lat = item["observation.latitude"][-1]
        #goal_lon = item["observation.longitude"][-1]
        #goal_compass = item["observation.compass_heading"][-1]      
        #goal_map_image = self._image_transforms(self.transform_PIL_tensor(self.map_image_gen.get_map_view(goal_lat, goal_lon, goal_compass)), flip_tf)
        #print("compass", current_compass, goal_compass)    
        
                                        
        #print(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)
        if flip_tf:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]        
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)
            
            x_loc = relative_mat[0,2]
            y_loc = relative_mat[1,2]
            yaw_loc = np.arctan2(relative_mat[1,0], relative_mat[1,1])
            #print(yaw_loc, goal_heading-current_heading)            
            #filt_goal = item["observation.filtered_position"][-1, None]
            #utm_goal = item["observation.utm_position"][-1, None]
            #let_goal = item["observation.utm_zone_letter"][-1, None]
            #num_goal = item["observation.utm_zone_number"][-1, None]
            #print("dataset", filt_goal, utm_goal, let_goal, num_goal)
            
            new_lat, new_lon, new_heading = self._transform_position(current_lat.item(), current_lon.item(), current_compass.item(), x_loc.item(), y_loc.item(), yaw_loc.item())              
            """
            try:
                goal_map_PIL = self.transform_PIL_tensor(self.map_image_gen.get_map_view(new_lat, new_lon, new_heading))
                goal_map_image = self._image_transforms(goal_map_PIL, flip_tf)
                goal_map_PILimage = to_pil_image(self._image_224(goal_map_PIL, flip_tf))                
            except Exception as e:
                goal_lat = item["observation.latitude"][-1]
                goal_lon = item["observation.longitude"][-1]
                #goal_compass = item["observation.compass_heading"][-1] 
                goal_compass = item["observation.filtered_heading"][-1] 
                goal_map_PIL = self.transform_PIL_tensor(self.map_image_gen.get_map_view(goal_lat, goal_lon, goal_compass))
                goal_map_image = self._image_transforms(goal_map_PIL, flip_tf)      
                goal_map_PILimage = to_pil_image(self._image_224(goal_map_PIL, flip_tf))              
            """
            #print("before", goal_pos_relative, relative_mat)            
            goal_pos_relative[1] *= -1                
            relative_mat[0,1] *= -1
            relative_mat[1,0] *= -1                                    
            #print("after", goal_pos_relative, goal_dist_gps)
        else:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)        

            x_loc = relative_mat[0,2]
            y_loc = relative_mat[1,2]
            yaw_loc = np.arctan2(relative_mat[1,0], relative_mat[1,1])
            #print(yaw_loc, goal_heading-current_heading)
            
            new_lat, new_lon, new_heading = self._transform_position(current_lat.item(), current_lon.item(), current_compass.item(), x_loc.item(), y_loc.item(), yaw_loc.item()) 
            """       
            try:
                goal_map_PIL = self.transform_PIL_tensor(self.map_image_gen.get_map_view(new_lat, new_lon, new_heading))
                goal_map_image = self._image_transforms(goal_map_PIL, flip_tf)
                goal_map_PILimage = to_pil_image(self._image_224(goal_map_PIL, flip_tf))  
            except Exception as e:
                goal_lat = item["observation.latitude"][-1]
                goal_lon = item["observation.longitude"][-1]
                #goal_compass = item["observation.compass_heading"][-1] 
                goal_compass = item["observation.filtered_heading"][-1]      
                goal_map_PIL = self.transform_PIL_tensor(self.map_image_gen.get_map_view(goal_lat, goal_lon, goal_compass))           
                goal_map_image = self._image_transforms(goal_map_PIL, flip_tf)     
                goal_map_PILimage = to_pil_image(self._image_224(goal_map_PIL, flip_tf))  
            """
        action_IL = []
        metric_waypoint_spacing = 0.25 #normalization        
        goal_pos_relative = goal_pos_relative/metric_waypoint_spacing #normalization        
        for i_traj in range(control_horizon-1):
            traj_relative_mat = to_local_coords_yaw(item["observation.filtered_position"][i_traj + 1, None], unnorm_position[0], current_heading, item["observation.filtered_heading"][i_traj + 1])    
            
            if flip_tf:
                #print("FLIP")
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, -traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], -traj_relative_mat[1,0]])
            else:
                #print("NO FLIP")
                action_IL.append([traj_relative_mat[0,2]/metric_waypoint_spacing, traj_relative_mat[1,2]/metric_waypoint_spacing, traj_relative_mat[1,1], traj_relative_mat[1,0]])
                            
        action_IL = torch.tensor(action_IL)        
        #print(action_IL.size())
                
        # action = einops.reduce(item["action"][:-1], "(a s) d -> a d", reduction="mean", s=self.action_spacing)
        #action_is_pad = einops.reduce(item["action_is_pad"][:-1], "(a s) -> a", reduction="max", s=self.action_spacing)
        heading = item["observation.filtered_heading"][:-1]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        
        #print("action", heading, np.unwrap(heading), np.diff(np.unwrap(heading)))
        action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
        action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
        action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing
        
        #action = [0.0]
        
        future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)
        which_dataset = 0

        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")
        image_flattened_crop = einops.rearrange(image_obs_crop, "... t c h w -> ... (t c) h w")
        #image_flattened = [0.0]
        #image_goal = [0.0]
        #image_current = [0.0]
        #goal_dist = [0.0]
        #goal_pos_relative =                obs_image, 

        #relative_mat = [0.0]        
        #goal_heading = 0.0
        #current_heading = 0.0
        #which_dataset = [0.0]
        #future_positions_unfiltered = [0.0]
        #idx = [0.0]
        #image_raw = [0.0]

        if goal_dist == 0:
            goal_is_negative = True
        else:
            goal_is_negative = False
        #[TODO] in GNM dataset, we set goal_dist == 20 when goal_dist is zero (current frame == goal frame) and we give random goal image. But we do in training code. We need to fix this not to confuse the users.
        
        
        action_mask = (
            (goal_dist < self.max_action_distance) and
            (goal_dist > self.min_action_distance) and
            (not goal_is_negative)
        )
        """
        #print(action_mask, goal_is_negative, goal_dist, self.max_action_distance, self.min_action_distance)
        return (
            torch.as_tensor(image_flattened, dtype=torch.float32),
            torch.as_tensor(image_flattened_crop, dtype=torch.float32),
            torch.as_tensor(image_goal, dtype=torch.float32),
            torch.as_tensor(image_goal2, dtype=torch.float32),            
            torch.as_tensor(image_current, dtype=torch.float32),            
            torch.as_tensor(action_IL, dtype=torch.float32),
            #torch.as_tensor(goal_dist, dtype=torch.int64),
            torch.as_tensor(goal_dist_gps, dtype=torch.int64),            
            torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            torch.as_tensor(relative_mat, dtype=torch.float32),  
            torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            torch.as_tensor(which_dataset, dtype=torch.int64),
            #torch.as_tensor(~action_is_pad, dtype=torch.float32),
            torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.float32),
            torch.as_tensor(action_mask, dtype=torch.float32),           
            #torch.as_tensor(image_raw, dtype=torch.float32),     
            #torch.as_tensor(ped_local_slice, dtype=torch.float32),         
            #torch.as_tensor(ped_local_slice_raw, dtype=torch.float32),        
            torch.as_tensor(current_map_image, dtype=torch.float32),         
            torch.as_tensor(goal_map_image, dtype=torch.float32),                 
            torch.as_tensor(ped_list_no_trans, dtype=torch.float32),
            torch.as_tensor(robot_local_slice, dtype=torch.float32),                                   
        )  
        """
        #relative_mat[1,1].unsqueeze(1), relative_mat[:,1,0].unsqueeze(1)
        obj_pose_norm = np.array((0.0, 0.0)) #dummy obj pose
        
        goal_pos = np.array(goal_pos_relative)        # Ensures goal_pos supports slicing like [0:1]
        #goal_yaw = np.array([goal_yaw])   
        #print(goal_pos[0:1].shape, np.array(relative_mat[1,1]).shape)
        #goal_pose_cos_sin = np.concatenate((goal_pos[0:1]*metric_waypoint_spacing, goal_pos[1:2]*metric_waypoint_spacing, np.array(relative_mat[1,1]), np.array(relative_mat[1,0]))), axis=0) #Adapting ViNT style action commands (X, Y, cos, sin)            
        #print(goal_pose_cos_sin.shape)
        #goal_pose_cos_sin = np.concatenate((goal_pos[0:1]*metric_waypoint_spacing, goal_pos[1:2]*metric_waypoint_spacing, np.array([relative_mat[1,1]]), np.array([relative_mat[1,0]])), axis=0)
        goal_pose_cos_sin = np.concatenate((goal_pos[0:1], goal_pos[1:2], np.array([relative_mat[1,1]]), np.array([relative_mat[1,0]])), axis=0)
          
        ### Adapting OpenVLA stle ###
        actions = action_IL
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        #yaw = np.arctan2(np.array([relative_mat[1,0]]), np.array([relative_mat[1,1]]))
        try:
            #lang = "move toward " + "[" + str(goal_pos[0:1]*metric_waypoint_spacing) + ", " + str(goal_pos[1:2]*metric_waypoint_spacing) + ", " + str(yaw) + "] on the current robot coordinate [X, Y, Yaw]."# I am giving X, Y and yaw angle." #dummy language prompt
            lang = "XXXXX"
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
        
        #print("tokernization length", len(labels))
        #print("check!!", labels.size(), input_ids.size())
        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        
        #print("frod", image_obs_topil.size(), image_obs_topil.max(), image_obs_topil.min(), image_goal_topil.size(), image_goal_topil.max(), image_goal_topil.min())
        pil_image_obs = to_pil_image(image_obs_topil)
        pil_image_goal = to_pil_image(image_goal_topil)        
        #dummy_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        #dummy_PIL = Image.fromarray(dummy_array)
        #pixel_values = self.image_transform(dummy_PIL)
        pixel_values = self.image_transform(pil_image_obs)
        pixel_values_wrist = self.image_transform(pil_image_goal)
        
        dummy_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        dummy_PIL = Image.fromarray(dummy_array)
        pixel_values_dummy = self.image_transform(dummy_PIL.transpose(Image.FLIP_LEFT_RIGHT))
        #pixel_values_curmap = self.image_transform(current_map_PILimage)
        #pixel_values_goalmap = self.image_transform(goal_map_PILimage)
                
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX
        dataset_name = "frod"

        obs_images_list = torch.split(image_flattened, 3, dim=0)
        cur_image_large = TF.resize(obs_images_list[-1], (224, 224))

        return_dict = dict(pixel_values=pixel_values, pixel_values_wrist=pixel_values_wrist, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=torch.as_tensor(actions), actions_nomad=torch.as_tensor(np.asarray(np.random.rand(8, ACTION_DIM), dtype=np.float32)), goal_pose=goal_pose_cos_sin, obj_pose_norm=obj_pose_norm, img_PIL=pil_image_obs, cur_image_crop = image_flattened_crop, cur_image = image_flattened, cur_image_large = cur_image_large, goal_image_crop=image_goal2, goal_image_8=image_goal, temp_dist=goal_dist_gps, cur_map_image=cur_image_large, goal_map_image=cur_image_large, pixel_values_curmap=pixel_values_dummy, pixel_values_goalmap=pixel_values_dummy)
        #return_dict = dict(pixel_values=pixel_values, pixel_values_wrist=pixel_values_wrist, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=torch.as_tensor(actions), actions_nomad=torch.as_tensor(actions), goal_pose=goal_pose_cos_sin, obj_pose_norm=obj_pose_norm, img_PIL=pil_image_obs, cur_image_crop = image_flattened_crop, cur_image = image_flattened, cur_image_large = cur_image_large, goal_image_crop=image_goal2, goal_image_8=image_goal, temp_dist=goal_dist_gps, cur_map_image=current_map_image, goal_map_image=goal_map_image, pixel_values_curmap=pixel_values_curmap, pixel_values_goalmap=pixel_values_goalmap)
        #return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=torch.as_tensor(actions), goal_pose=goal_pose_cos_sin, obj_pose_norm=obj_pose_norm, img_PIL=dummy_PIL, cur_image_crop = image_obs_crop_r, cur_image = image_obs_r, cur_image_large = cur_image_r, goal_image_crop=goal_image_full_r, goal_image_8=goal_image_full_8_r, temp_dist=goal_id)
        return return_dict   

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = to_indices[self.dataset_cache["episode_index"]]

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)

        # heading = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        # initial_heading = heading[indices]
        # final_heading = heading[target_indices]

        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        return torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class EpisodeSampler_IL_MMN(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        episode_index_from: int,
        episode_index_to: int,
        goal_horizon: int,
        data_split_type: str,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
    ):   
        self.dataset = dataset
        self.goal_horizon = goal_horizon
        self.shuffle = shuffle

        self.num_replicas = num_replicas if num_replicas is not None else torch.distributed.get_world_size()
        self.rank = rank if rank is not None else torch.distributed.get_rank()
        self.epoch = 0

        from_idx = dataset.episode_data_index["from"][episode_index_from].item()
        to_idx = dataset.episode_data_index["to"][episode_index_to].item()
        self.frame_ids_range = list(range(from_idx, to_idx))

        print(f"[Rank {self.rank}] from_idx {from_idx}, to_idx {to_idx}")  

        if data_split_type == "train":
            with open('./prismatic/vla/datasets/sampler/train_yaw_small.pkl', 'rb') as file:
                data = pickle.load(file)
            with open('./prismatic/vla/datasets/sampler/train_ped_fix.pkl', 'rb') as file:
                data_ped = pickle.load(file)                
        elif data_split_type == "test":
            with open('./prismatic/vla/datasets/sampler/test_yaw_small.pkl', 'rb') as file:
                data = pickle.load(file)
            with open('./prismatic/vla/datasets/sampler/test_ped_fix.pkl', 'rb') as file:
                data_ped = pickle.load(file)         

        self.yaw_list = data[1]
        self.ped_list = data_ped[1]        
        self.init_idx = data[0][0]                        

        self.total_size = math.ceil(len(self.frame_ids_range) / self.num_replicas) * self.num_replicas
        self.num_samples = self.total_size // self.num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator:
        g = torch.Generator()
        g.manual_seed(self.epoch + self.rank)  # ensure different processes shuffle deterministically but differently

        indices_new = []

        for idx in tqdm(self.frame_ids_range):
            thres_rate = random.random()

            if self.yaw_list[idx - self.init_idx] % (2 * 3.14) > 3.14:
                ang_yaw = self.yaw_list[idx - self.init_idx] % (2 * 3.14) - 2.0 * 3.14
            else:
                ang_yaw = self.yaw_list[idx - self.init_idx] % (2 * 3.14)

            while abs(ang_yaw) > 2.0:
                idx = random.choice(self.frame_ids_range)
                if self.yaw_list[idx - self.init_idx] % (2 * 3.14) > 3.14:
                    ang_yaw = self.yaw_list[idx - self.init_idx] % (2 * 3.14) - 2.0 * 3.14
                else:
                    ang_yaw = self.yaw_list[idx - self.init_idx] % (2 * 3.14)

            if thres_rate < 0.5:
                while not (abs(ang_yaw) > 0.4 and abs(ang_yaw) < 2.0):
                    idx = random.choice(self.frame_ids_range)
                    if self.yaw_list[idx - self.init_idx] % (2 * 3.14) > 3.14:
                        ang_yaw = self.yaw_list[idx - self.init_idx] % (2 * 3.14) - 2.0 * 3.14
                    else:
                        ang_yaw = self.yaw_list[idx - self.init_idx] % (2 * 3.14)

            indices_new.append(idx)

        # Optional shuffling
        if self.shuffle:
            indices_new = [indices_new[i] for i in torch.randperm(len(indices_new), generator=g)]

        # Pad to be evenly divisible across GPUs
        if len(indices_new) < self.total_size:
            extra = indices_new[: self.total_size - len(indices_new)]
            indices_new += extra
        assert len(indices_new) == self.total_size

        # Subsample for this process
        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(indices_new[start:end])

    def __len__(self) -> int:
        return self.num_samples

"""
class EpisodeSampler_IL_MMN(torch.utils.data.Sampler):
    def __init__(self, dataset: ViNTLeRobotDataset, episode_index_from: int, episode_index_to: int, goal_horizon: int, data_split_type: str):
        self.dataset = dataset
        self.goal_horizon = goal_horizon
        from_idx = dataset.episode_data_index["from"][episode_index_from].item()
        to_idx = dataset.episode_data_index["to"][episode_index_to].item()
        self.frame_ids_range = range(from_idx, to_idx)
        print("from_idx", from_idx, "to_idx", to_idx)  

        if data_split_type == "train":
            with open('./prismatic/vla/datasets/sampler/train_yaw_small.pkl', 'rb') as file:
                data = pickle.load(file)
            with open('./prismatic/vla/datasets/sampler/train_ped_fix.pkl', 'rb') as file:
                data_ped = pickle.load(file)                
        elif data_split_type == "test":
            with open('./prismatic/vla/datasets/sampler/test_yaw_small.pkl', 'rb') as file:
                data = pickle.load(file)
            with open('./prismatic/vla/datasets/sampler/test_ped_fix.pkl', 'rb') as file:
                data_ped = pickle.load(file)         
     
        self.yaw_list = data[1]
        self.ped_list = data_ped[1]        
        self.init_idx = data[0][0]                        
                                      
    def __iter__(self) -> Iterator:   
        indices_new = []
        yawangle_list = []
        
        for idx in tqdm(self.frame_ids_range):
            
            thres_rate = random.random()
            if self.yaw_list[idx-self.init_idx] % (2*3.14) > 3.14:
                ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14) - 2.0*3.14
            else:
                ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14)   
            
            while abs(ang_yaw) > 2.0:
                idx = random.choice(self.frame_ids_range)   
                if self.yaw_list[idx-self.init_idx] % (2*3.14) > 3.14:
                    ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14) - 2.0*3.14
                else:
                    ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14)   
            
            #if thres_rate < 0.8:
            #    while not ((abs(ang_yaw) > 0.4 or self.ped_list[idx-self.init_idx] != 10000.0) and abs(ang_yaw) < 2.0): 
            if thres_rate < 0.5:                  
                while not (abs(ang_yaw) > 0.4 and abs(ang_yaw) < 2.0):                 
                    idx = randoFMMNm.choice(self.frame_ids_range)            
                    if self.yaw_list[idx-self.init_idx] % (2*3.14) > 3.14:
                        ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14) - 2.0*3.14
                    else:
                        ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14)             
            
            indices_new.append(idx)           

        indices_new_random = random.sample(indices_new, len(indices_new)) 
        return iter(indices_new_random)

    def __len__(self) -> int:
        return len(self.frame_ids_range)   
"""
class EpisodeSampler_IL(torch.utils.data.Sampler):
    def __init__(self, dataset: ViNTLeRobotDataset, episode_index_from: int, episode_index_to: int, goal_horizon: int, data_split_type: str):
        self.dataset = dataset
        self.goal_horizon = goal_horizon
        from_idx = dataset.episode_data_index["from"][episode_index_from].item()
        to_idx = dataset.episode_data_index["to"][episode_index_to].item()
        self.frame_ids_range = range(from_idx, to_idx)
        print("from_idx", from_idx, "to_idx", to_idx)  

        if data_split_type == "train":
            with open('./vint_train/data/sampler/train_yaw_small.pkl', 'rb') as file:
                data = pickle.load(file)
            with open('./vint_train/data/sampler/train_ped_fix.pkl', 'rb') as file:
                data_ped = pickle.load(file)                
        elif data_split_type == "test":
            with open('./vint_train/data/sampler/test_yaw_small.pkl', 'rb') as file:
                data = pickle.load(file)
            with open('./vint_train/data/sampler/test_ped_fix.pkl', 'rb') as file:
                data_ped = pickle.load(file)         
     
        self.yaw_list = data[1]
        self.ped_list = data_ped[1]        
        self.init_idx = data[0][0]                        
                                      
    def __iter__(self) -> Iterator:   
        indices_new = []
        yawangle_list = []
        
        for idx in tqdm(self.frame_ids_range):
            
            thres_rate = random.random()
            if self.yaw_list[idx-self.init_idx] % (2*3.14) > 3.14:
                ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14) - 2.0*3.14
            else:
                ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14)   
            
            while abs(ang_yaw) > 2.0:
                idx = random.choice(self.frame_ids_range)   
                if self.yaw_list[idx-self.init_idx] % (2*3.14) > 3.14:
                    ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14) - 2.0*3.14
                else:
                    ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14)   
            
            #if thres_rate < 0.8:
            #    while not ((abs(ang_yaw) > 0.4 or self.ped_list[idx-self.init_idx] != 10000.0) and abs(ang_yaw) < 2.0): 
            if thres_rate < 0.5:                  
                while not (abs(ang_yaw) > 0.4 and abs(ang_yaw) < 2.0):                 
                    idx = random.choice(self.frame_ids_range)            
                    if self.yaw_list[idx-self.init_idx] % (2*3.14) > 3.14:
                        ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14) - 2.0*3.14
                    else:
                        ang_yaw = self.yaw_list[idx-self.init_idx] % (2*3.14)             
            
            indices_new.append(idx)           

        indices_new_random = random.sample(indices_new, len(indices_new)) 
        return iter(indices_new_random)

    def __len__(self) -> int:
        return len(self.frame_ids_range)   
           
class EpisodeSampler_annotate(torch.utils.data.Sampler):
    def __init__(self, dataset: ViNTLeRobotDataset, episode_index_from: int, episode_index_to: int, goal_horizon: int, data_split_type: str):
        self.dataset = dataset
        self.goal_horizon = goal_horizon
        from_idx = dataset.episode_data_index["from"][episode_index_from].item()
        to_idx = dataset.episode_data_index["to"][episode_index_to].item()
        self.frame_ids = range(from_idx, to_idx)
        """
        if data_split_type == "train":
            with open('/media/noriaki/Noriaki_Data2/learning-language-navigation/train/vint_train/training/train_yaw_small.pkl', 'rb') as file:
                data = pickle.load(file)
            self.frame_ids = data[0][from_idx:to_idx]
            self.yaw_list = data[1][from_idx:to_idx]
                        
        elif data_split_type == "test":
            with open('/media/noriaki/Noriaki_Data2/learning-language-navigation/train/vint_train/training/test_yaw_small.pkl', 'rb') as file:
                data = pickle.load(file)
            
            to_idx_test = data[0].index(to_idx-1)    
            from_idx_test = data[0].index(from_idx)           
            self.frame_ids = data[0][from_idx_test:to_idx_test]
            self.yaw_list = data[1][from_idx_test:to_idx_test]
        """
        print(data_split_type, from_idx, to_idx)
            
        #self.init_idx = data[0][0]
                                      
    def __iter__(self) -> Iterator:
        indices = self.frame_ids
        indices_new = []
        yawangle_list = []
        
        for idx in tqdm(indices):                      
            indices_new.append(idx)           

        #print(indices_new)
        #indices_new_random = random.sample(indices_new, len(indices_new)) 
        return iter(indices_new)

    def __len__(self) -> int:
        return len(self.frame_ids)              

class EpisodeSampler_annotate_10k(torch.utils.data.Sampler):
    def __init__(self, dataset: ViNTDataset_annotate_10k, episode_index_from: int, episode_index_to: int, goal_horizon: int, data_split_type: str):
        self.dataset = dataset
        self.goal_horizon = goal_horizon
        from_idx = dataset.episode_data_index["from"][episode_index_from].item()
        to_idx = dataset.episode_data_index["to"][episode_index_to].item()
        self.frame_ids = range(from_idx, to_idx)
        """
        if data_split_type == "train":
            with open('/media/noriaki/Noriaki_Data2/learning-language-navigation/train/vint_train/training/train_yaw_small.pkl', 'rb') as file:
                data = pickle.load(file)
            self.frame_ids = data[0][from_idx:to_idx]
            self.yaw_list = data[1][from_idx:to_idx]
                        
        elif data_split_type == "test":
            with open('/media/noriaki/Noriaki_Data2/learning-language-navigation/train/vint_train/training/test_yaw_small.pkl', 'rb') as file:
                data = pickle.load(file)
            
            to_idx_test = data[0].index(to_idx-1)    
            from_idx_test = data[0].index(from_idx)           
            self.frame_ids = data[0][from_idx_test:to_idx_test]
            self.yaw_list = data[1][from_idx_test:to_idx_test]
        """
        print(data_split_type, from_idx, to_idx)
            
        #self.init_idx = data[0][0]
                                      
    def __iter__(self) -> Iterator:
        indices = self.frame_ids
        indices_new = []
        yawangle_list = []
        
        for idx in tqdm(indices):                      
            indices_new.append(idx)           

        #print(indices_new)
        #indices_new_random = random.sample(indices_new, len(indices_new)) 
        return iter(indices_new)

    def __len__(self) -> int:
        return len(self.frame_ids)    
 
class EpisodeSampler_IL_10k(torch.utils.data.Sampler):
    def __init__(self, dataset: ViNTDataset_10k, episode_index_from: int, episode_index_to: int, goal_horizon: int, data_split_type: str):
        self.dataset = dataset
        #self.goal_horizon = goal_horizon
        #print("episode number", len(dataset.episode_data_index["from"]))
        #from_idx = dataset.episode_data_index["from"][episode_index_from].item()
        #to_idx = dataset.episode_data_index["to"][episode_index_to].item()
        
        if data_split_type == "train":
            yaw_small = np.load("/mnt/ephemeral2/noriaki/frodobots_dataset/yaw_10k_small.npy")
            from_idx = int(0)
            to_idx = int(164748770*0.95)
            self.yaw_list = yaw_small[from_idx:to_idx]
                        
        elif data_split_type == "test":
            yaw_small = np.load("/mnt/ephemeral2/noriaki/frodobots_dataset/yaw_10k_small.npy")
            from_idx = int(164748770*0.95)
            to_idx = int(164748770)
            self.yaw_list = yaw_small[from_idx:to_idx]
        
        self.frame_ids_range = range(from_idx, to_idx)
        self.init_idx = self.frame_ids_range[0]
        print(data_split_type, from_idx, to_idx)

    def __iter__(self) -> Iterator:
        indices_new = []
        yawangle_list = []
        
        for idx in tqdm(self.frame_ids_range):            
            thres_rate = random.random()

            ang_yaw = self.yaw_list[idx-self.init_idx] 
            while abs(ang_yaw) > 2.0:
                idx = random.choice(self.frame_ids_range)     
                ang_yaw = self.yaw_list[idx-self.init_idx]
            if thres_rate < 0.5:
                while not (abs(ang_yaw) > 0.4 and abs(ang_yaw) < 2.0):                  
                    idx = random.choice(self.frame_ids_range)            
                    ang_yaw = self.yaw_list[idx-self.init_idx]            
            
            indices_new.append(idx)           
        """
        for idx in tqdm(self.frame_ids_range):                      
            indices_new.append(idx) 
        """    
        indices_new_random = random.sample(indices_new, len(indices_new)) 
        return iter(indices_new_random)

    def __len__(self) -> int:
        return len(self.frame_ids_range)   
 
