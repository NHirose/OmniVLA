import random
import math
import pickle
from pathlib import Path
from typing import Tuple, Callable, Union, Any, Dict, List, Optional, Type, Iterator

import numpy as np
import zarr
import utm
from tqdm import tqdm
from matplotlib.path import Path as Pathmat
from PIL import Image

import torch
import torch.utils.data
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import einops

from torch.utils.data import Dataset
from itertools import accumulate

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import load_previous_and_future_frames
from lerobot.common.datasets.video_utils import load_from_videos

from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    STOP_INDEX,
)
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from transformers import PreTrainedTokenizerBase
from prismatic.models.backbones.vision import ImageTransform


def yaw_rotmat(yaw: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Return 3x3 rotation matrix (or torch tensor) for a yaw angle."""
    if isinstance(yaw, torch.Tensor):
        return torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw)],
                [torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)],
                [torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)],
            ]
        )
    else:
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )


def trans_mat(
    pos: float | np.ndarray | torch.Tensor, yaw: float | np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """Return homogeneous transform matrix for position and yaw."""
    if isinstance(yaw, torch.Tensor):
        return torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), pos[0]],
                [torch.sin(yaw), torch.cos(yaw), pos[1]],
                [torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)],
            ]
        )
    else:
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), pos[0]],
                [np.sin(yaw), np.cos(yaw), pos[1]],
                [0.0, 0.0, 1.0],
            ]
        )


def to_local_coords(
    positions: np.ndarray | torch.Tensor,
    curr_pos: np.ndarray | torch.Tensor,
    curr_yaw: float | np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    Convert positions to local coordinates relative to curr_pos and curr_yaw.

    positions: (..., 2) or (..., 3)
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rot2 = rotmat[:2, :2]
        return (positions - curr_pos) @ rot2
    elif positions.shape[-1] == 3:
        return (positions - curr_pos) @ rotmat
    else:
        raise ValueError("positions must have last dim 2 or 3")


def to_local_coords_yaw(
    positions: np.ndarray | torch.Tensor,
    curr_pos: np.ndarray | torch.Tensor,
    curr_yaw: float | np.ndarray | torch.Tensor,
    goal_yaw: float | np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    Return relative transform matrix between current frame (curr_pos, curr_yaw)
    and goal frame defined by positions[0] and goal_yaw.
    """
    cur_mat = trans_mat(curr_pos, curr_yaw)
    goal_mat = trans_mat(positions[0], goal_yaw)
    cur_mat_inv = torch.linalg.inv(cur_mat) if isinstance(cur_mat, torch.Tensor) else np.linalg.inv(cur_mat)
    return cur_mat_inv @ goal_mat


class ActionFormat:
    WAYPOINT = 1
    WAYPOINT_ANGLE = 2
    LINEAR_ANGULAR = 3

    @staticmethod
    def from_str(s: str) -> int:
        s = s.strip().lower()
        if s == "waypoint":
            return ActionFormat.WAYPOINT
        if s == "waypoint_angle":
            return ActionFormat.WAYPOINT_ANGLE
        if s == "linear_angular":
            return ActionFormat.LINEAR_ANGULAR
        raise ValueError(f"Unknown action format {s}")


def load_pickle(
    dataset: zarr.Array,
    index: int,
    episode_data_index: dict[str, torch.Tensor],
    delta_timestamps: dict[str, list[float]],
) -> dict[torch.Tensor]:
    """
    Helper (kept small) — compute nearest data ids for provided delta_timestamps.
    Returns array of dataset indices (numpy).
    """
    ep_id = dataset["episode_index"][index].item()
    ep_data_id_from = episode_data_index["from"][ep_id].item()
    ep_data_id_to = episode_data_index["to"][ep_id].item()
    ep_data_ids = torch.arange(ep_data_id_from, ep_data_id_to, 1)

    data_ids = {}
    for key, delta_ts in delta_timestamps.items():
        current_ts = dataset["timestamp"][index]
        query_ts = current_ts + torch.tensor(delta_ts)
        ep_timestamps = torch.from_numpy(dataset["timestamp"][ep_data_id_from:ep_data_id_to]).float()
        dist = torch.cdist(query_ts[:, None], ep_timestamps[:, None], p=1)
        _, argmin_ = dist.min(1)
        data_ids[key] = ep_data_ids[argmin_].numpy()
    return data_ids


def load_frames_zarr(
    dataset: zarr.Array,
    index: int,
    episode_data_index: dict[str, torch.Tensor],
    delta_timestamps: dict[str, list[float]],
    tolerance_s: float,
) -> dict:
    """
    Given a zarr dataset and an index, fetch observations / frames for requested delta timestamps.
    Returns dictionary with keys for each requested field and *_is_pad booleans where relevant.
    """
    ep_id = dataset["episode_index"][index].item()
    ep_data_id_from = episode_data_index["from"][ep_id].item()
    ep_data_id_to = episode_data_index["to"][ep_id].item()
    ep_data_ids = torch.arange(ep_data_id_from, ep_data_id_to, 1)

    ep_timestamps = torch.from_numpy(dataset["timestamp"][ep_data_id_from:ep_data_id_to]).float()
    ep_first_ts = ep_timestamps[0]
    ep_last_ts = ep_timestamps[-1]
    current_ts = dataset["timestamp"][index]

    item: Dict[str, Any] = {}
    for key, delta_ts in delta_timestamps.items():
        timestamp_key = f"{key}.timestamp"
        path_key = f"{key}.path"
        is_video = timestamp_key in dataset.keys() and path_key in dataset.keys()

        if delta_ts is None:
            if key in dataset.keys():
                item[key] = torch.from_numpy(np.asarray(dataset[key][index]))
            elif is_video:
                item[key] = [{"path": dataset[path_key][i.item()], "timestamp": dataset[timestamp_key][i.item()]} for i in ep_data_ids]
            else:
                raise ValueError(f"Timestamp key {timestamp_key} not found in dataset")
        else:
            query_ts = current_ts + torch.tensor(delta_ts)
            dist = torch.cdist(query_ts[:, None], ep_timestamps[:, None], p=1)
            min_, argmin_ = dist.min(1)

            is_pad = min_ > tolerance_s
            data_ids = ep_data_ids[argmin_].numpy()

            if is_video:
                item[key] = [{"path": dataset[path_key][i], "timestamp": float(dataset[timestamp_key][i])} for i in data_ids]
            else:
                item[key] = torch.from_numpy(dataset[key][data_ids])

            item[f"{key}_is_pad"] = is_pad

    return item


class Frodobots_Dataset(LeRobotDataset):
    def __init__(
        self,
        action_tokenizer: PreTrainedTokenizerBase,
        base_tokenizer: ActionTokenizer,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        video: str,
        root: Path | None,
        predict_stop_token: bool = True,
        split: str = "train",
        action_format: Union[ActionFormat, str] = ActionFormat.WAYPOINT,
        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        goal_horizon2: int = 20,
        context_size: int = 5,
        context_spacing: int = 1,
        dataset_framerate: int = 10,
        image_size: Tuple[int, int] = (120, 160),
        image_transforms: Optional[Callable] = None,
    ):
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.goal_horizon2 = goal_horizon2
        self.context_size = context_size
        self.context_spacing = context_spacing
        self.image_size = image_size

        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.prompt_builder = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.image_transform = image_transform
        self.image_transforms = image_transforms

        print("Frodobots Dataset path", root)
        super().__init__(
            repo_id="frodobots_dataset",
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
                "action": [i * action_spacing * self.dt for i in range(action_horizon)],
            },
        )

        # dataset cache stored as zarr array -> keep as numpy arrays for speed
        self.dataset_cache = zarr.load(Path(root) / "frodobots_dataset" / "dataset_cache.zarr")
        self.dataset_cache = {k: np.asarray(v) for k, v in self.dataset_cache.items()}

        self.min_action_distance = 3
        self.max_action_distance = 20

        self.transform_PIL_tensor = transforms.ToTensor()

    def _image_transforms(self, img: torch.Tensor, flip: bool) -> torch.Tensor:
        """Resize to target image size and optionally horizontally flip."""
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        img = TF.resize(img, self.image_size)

        if flip:
            img = torch.flip(img, dims=(-1,))

        return img

    def _image_transforms_rand_crop(self, img: torch.Tensor, flip: bool) -> torch.Tensor:
        """Random crop with constrained random offsets (keeps runtime deterministic-ish)."""
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        top = np.random.randint(0, 20)
        height = 144 - np.random.randint(0, 20) - top
        left = np.random.randint(0, 25)
        width = 256 - 2 * left

        img = TF.crop(img, int(top), int(left), int(height), int(width))
        img = TF.resize(img, self.image_size)

        if flip:
            img = torch.flip(img, dims=(-1,))

        return img

    def _image_rand_crop_224(self, img: torch.Tensor, flip: bool) -> torch.Tensor:
        """Produce a 224x224 random crop variant for visualization/consistency."""
        top = np.random.randint(0, 20)
        height = 144 - np.random.randint(0, 20) - top
        left = np.random.randint(0, 25)
        width = 256 - 2 * left

        img = TF.crop(img, int(top), int(left), int(height), int(width))
        img = TF.resize(img, (224, 224))

        if flip:
            img = torch.flip(img, dims=(-1,))

        return img

    def _image_224(self, img: torch.Tensor, flip: bool) -> torch.Tensor:
        img = TF.resize(img, (224, 224))
        if flip:
            img = torch.flip(img, dims=(-1,))
        return img

    def _image_transforms_depth(self, img: torch.Tensor, flip: bool):
        """Return resized depth-like and rgb tensors used by some models."""
        img_rsize = TF.resize(img, (128, 416))
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert action sequences to positions for visualization (depends on action format)."""
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
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
        easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
        return easting, northing, zone_number, zone_letter

    def _utm_to_latlon(self, easting, northing, zone_number, zone_letter):
        lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)
        return lat, lon

    def _transform_position(self, lat, lon, heading, X, Y, theta):
        """
        Compute new lat/lon and heading after moving in local (X forward, Y left) + yaw theta.
        Uses UTM conversions and returns (lat, lon, heading).
        """
        easting, northing, zone_number, zone_letter = self._latlon_to_utm(lat, lon)
        heading_rad = heading
        new_heading = heading - theta

        delta_easting = np.sqrt(X**2 + Y**2) * np.sin(new_heading)
        delta_northing = np.sqrt(X**2 + Y**2) * np.cos(new_heading)

        new_easting = easting + delta_easting
        new_northing = northing + delta_northing

        new_lat, new_lon = self._utm_to_latlon(new_easting, new_northing, zone_number, zone_letter)
        new_heading = heading - theta

        return new_lat, new_lon, new_heading

    def _add_shadow_to_tensor_image(self, img_tensor: torch.Tensor, num_shadows: int = 1, shadow_intensity: float = 0.6) -> torch.Tensor:
        """
        Add polygonal shadow(s) to a tensor image or stack of images using matplotlib Path.
        Accepts both a single image (3, H, W) or multi-frame tensor (T, C, H, W).
        """
        def polygon_to_mask(points, H, W):
            yy, xx = np.mgrid[:H, :W]
            coords = np.stack((xx.ravel(), yy.ravel()), axis=-1)
            path = Pathmat(points)
            mask_np = path.contains_points(coords).reshape(H, W)
            return torch.from_numpy(mask_np.astype(np.float32))

        # handle batched sequence of frames (T, C, H, W) or single image (3, H, W)
        if img_tensor.ndim == 4:  # (T, C, H, W)
            shadow_img_list = []
            for i in range(img_tensor.shape[0]):
                C, H, W = img_tensor[i].shape
                shadow_mask = torch.zeros((H, W), dtype=torch.float32)
                for _ in range(num_shadows):
                    num_points = random.randint(3, 8)
                    points = [(random.randint(0, W - 1), random.randint(0, H - 1)) for _ in range(num_points)]
                    shadow_mask += polygon_to_mask(points, H, W)
                shadow_mask = torch.clamp(shadow_mask, 0, 1)
                shadow_factor = 1 - (shadow_mask * shadow_intensity)
                shadow_factor = shadow_factor.unsqueeze(0).expand_as(img_tensor[i])
                shadowed_img_c = img_tensor[i] * shadow_factor
                shadow_img_list.append(shadowed_img_c.unsqueeze(0))
            return torch.cat(shadow_img_list, dim=0)
        elif img_tensor.ndim == 3:  # (C, H, W)
            C, H, W = img_tensor.shape
            shadow_mask = torch.zeros((H, W), dtype=torch.float32)
            for _ in range(num_shadows):
                num_points = random.randint(3, 8)
                points = [(random.randint(0, W - 1), random.randint(0, H - 1)) for _ in range(num_points)]
                shadow_mask += polygon_to_mask(points, H, W)
            shadow_mask = torch.clamp(shadow_mask, 0, 1)
            shadow_factor = 1 - (shadow_mask * shadow_intensity)
            shadow_factor = shadow_factor.unsqueeze(0).expand_as(img_tensor)
            return img_tensor * shadow_factor
        else:
            raise ValueError("img_tensor must be (C,H,W) or (T,C,H,W)")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # sample distances and compute delta timestamps
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist2 = min(8, episode_length_remaining)
        goal_dist_gps = np.random.randint(0, min(self.goal_horizon2, episode_length_remaining))

        # prepare delta timestamps (add goals)
        delta_timestamps = self.delta_timestamps or {}
        delta_timestamps = {k: list(v) if v is not None else None for k, v in delta_timestamps.items()}
        # add goal offsets to every non-None entry
        extra = [goal_dist * self.dt * self.action_spacing, goal_dist2 * self.dt * self.action_spacing, goal_dist_gps * self.dt * self.action_spacing]
        delta_timestamps = {
            k: (v + extra if v is not None else None) for k, v in delta_timestamps.items()
        }
        # control horizon timestamps
        control_horizon = self.action_horizon + 1
        delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
            i * self.dt * self.action_spacing for i in range(control_horizon)
        ] + [goal_dist * self.dt * self.action_spacing] + [goal_dist_gps * self.dt * self.action_spacing]

        # load frames from zarr dataset
        item = load_frames_zarr(self.dataset_cache, idx, self.episode_data_index, delta_timestamps, self.tolerance_s)

        flip_tf = random.random() > 0.5
        num_shadows = random.randint(0, 3)
        shadow_intensity = 0.0  # currently disabled; set >0 to enable

        # load recent frames from videos
        image_obs_raw = load_from_videos(
            {"observation.images.front": item["observation.images.front"][:-3]},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"]

        image_obs_raw_s = self._add_shadow_to_tensor_image(image_obs_raw, num_shadows=num_shadows, shadow_intensity=shadow_intensity)
        image_obs_crop = self._image_transforms_rand_crop(image_obs_raw_s, flip_tf)
        image_obs = self._image_transforms(image_obs_raw, flip_tf)

        image_goal2 = self._image_transforms(
            load_from_videos(
                {"observation.images.front": item["observation.images.front"][-2]},
                ["observation.images.front"],
                self.videos_dir,
                self.tolerance_s,
                self.video_backend,
            )["observation.images.front"],
            flip_tf,
        )

        image_goal = self._image_transforms_rand_crop(
            self._add_shadow_to_tensor_image(
                load_from_videos(
                    {"observation.images.front": item["observation.images.front"][-1]},
                    ["observation.images.front"],
                    self.videos_dir,
                    self.tolerance_s,
                    self.video_backend,
                )["observation.images.front"],
                num_shadows=num_shadows,
                shadow_intensity=shadow_intensity,
            ),
            flip_tf,
        )

        image_goal_topil = self._image_rand_crop_224(
            self._add_shadow_to_tensor_image(
                load_from_videos(
                    {"observation.images.front": item["observation.images.front"][-1]},
                    ["observation.images.front"],
                    self.videos_dir,
                    self.tolerance_s,
                    self.video_backend,
                )["observation.images.front"],
                num_shadows=num_shadows,
                shadow_intensity=shadow_intensity,
            ),
            flip_tf,
        )

        image_current_shadow = self._add_shadow_to_tensor_image(
            load_from_videos(
                {"observation.images.front": item["observation.images.front"][-4]},
                ["observation.images.front"],
                self.videos_dir,
                self.tolerance_s,
                self.video_backend,
            )["observation.images.front"],
            num_shadows=num_shadows,
            shadow_intensity=shadow_intensity,
        )

        image_current, image_raw = self._image_transforms_depth(image_current_shadow, flip_tf)
        image_obs_topil = self._image_rand_crop_224(image_current_shadow, flip_tf)

        # pedestrian / robot placeholders (kept for compatibility)
        ped_list_no_trans = [0.0]
        ped_local_slice = [0.0]
        ped_local_slice_raw = [0.0]
        robot_local_slice = [0.0]

        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]
        goal_heading = item["observation.filtered_heading"][-1]

        current_lat = item["observation.latitude"][0]
        current_lon = item["observation.longitude"][0]
        current_compass = item["observation.filtered_heading"][0]

        # compute local goal pose and relative transform
        if flip_tf:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)

            x_loc = relative_mat[0, 2]
            y_loc = relative_mat[1, 2]
            yaw_loc = np.arctan2(relative_mat[1, 0], relative_mat[1, 1])

            new_lat, new_lon, new_heading = self._transform_position(
                current_lat.item(), current_lon.item(), current_compass.item(), x_loc.item(), y_loc.item(), yaw_loc.item()
            )

            goal_pos_relative[1] *= -1
            relative_mat[0, 1] *= -1
            relative_mat[1, 0] *= -1
        else:
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)

            x_loc = relative_mat[0, 2]
            y_loc = relative_mat[1, 2]
            yaw_loc = np.arctan2(relative_mat[1, 0], relative_mat[1, 1])

            new_lat, new_lon, new_heading = self._transform_position(
                current_lat.item(), current_lon.item(), current_compass.item(), x_loc.item(), y_loc.item(), yaw_loc.item()
            )

        # build IL-style actions for control horizon
        action_IL = []
        metric_waypoint_spacing = 0.25
        goal_pos_relative = goal_pos_relative / metric_waypoint_spacing

        for i_traj in range(control_horizon - 1):
            traj_relative_mat = to_local_coords_yaw(item["observation.filtered_position"][i_traj + 1, None], unnorm_position[0], current_heading, item["observation.filtered_heading"][i_traj + 1])
            if flip_tf:
                action_IL.append([traj_relative_mat[0, 2] / metric_waypoint_spacing, -traj_relative_mat[1, 2] / metric_waypoint_spacing, traj_relative_mat[1, 1], -traj_relative_mat[1, 0]])
            else:
                action_IL.append([traj_relative_mat[0, 2] / metric_waypoint_spacing, traj_relative_mat[1, 2] / metric_waypoint_spacing, traj_relative_mat[1, 1], traj_relative_mat[1, 0]])

        action_IL = torch.tensor(action_IL)
        heading = item["observation.filtered_heading"][:-1]
        direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)

        action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
        action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
        action = torch.stack([action_forward[: self.action_horizon], action_steer[: self.action_horizon]], dim=-1) / self.dt / self.action_spacing

        future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)
        which_dataset = 0

        image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")
        image_flattened_crop = einops.rearrange(image_obs_crop, "... t c h w -> ... (t c) h w")

        goal_is_negative = (goal_dist == 0)

        action_mask = (goal_dist < self.max_action_distance) and (goal_dist > self.min_action_distance) and (not goal_is_negative)

        obj_pose_norm = np.array((0.0, 0.0))
        goal_pos = np.array(goal_pos_relative)
        goal_pose_cos_sin = np.concatenate(
            (goal_pos[0:1], goal_pos[1:2], np.array([relative_mat[1, 1]]), np.array([relative_mat[1, 0]])), axis=0
        )

        # OpenVLA style tokenization & prompt
        actions = action_IL
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = "".join(self.action_tokenizer(future_actions))
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": "No language instruction"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        prompt_builder = self.prompt_builder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        pil_image_obs = to_pil_image(image_obs_topil)
        pil_image_goal = to_pil_image(image_goal_topil)
        pixel_values = self.image_transform(pil_image_obs)
        pixel_values_g = self.image_transform(pil_image_goal)

        # dummy image for augmentation consistency
        dummy_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        dummy_PIL = Image.fromarray(dummy_array)
        pixel_values_dummy = self.image_transform(dummy_PIL.transpose(Image.FLIP_LEFT_RIGHT))

        # mask out tokens we don't compute loss for
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        dataset_name = "frod"
        modality_list = [4, 5, 6]
        if goal_dist_gps <= 20:
            modality_id = random.choice(modality_list)
        else:
            modality_id = random.choice(modality_list[0:2])

        action_select_mask = torch.tensor(0.0)

        return dict(
            pixel_values=pixel_values,
            pixel_values_goal=pixel_values_g,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            modality_id=modality_id,
            actions=torch.as_tensor(actions),
            action_select_mask=action_select_mask,
            goal_pose=goal_pose_cos_sin,
            obj_pose_norm=obj_pose_norm,
            img_PIL=pil_image_obs,
            gimg_PIL=pil_image_goal,
            cur_image=image_flattened,
            goal_image_8=image_goal,
            temp_dist=goal_dist_gps,
            lan_prompt="No language instruction",
        )

    def get_sampler(self, base_rate: float = 0.1):
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


class EpisodeSampler_Frodobots(torch.utils.data.Sampler):
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

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("Distributed mode must be initialized before using this sampler.")

        self.num_replicas = num_replicas or torch.distributed.get_world_size()
        self.rank = rank or torch.distributed.get_rank()
        self.epoch = 0

        from_idx = dataset.episode_data_index["from"][episode_index_from].item()
        to_idx = dataset.episode_data_index["to"][episode_index_to].item()
        self.frame_ids_range = list(range(from_idx, to_idx))

        # Load split-specific metadata
        base_path = "./prismatic/vla/datasets/sampler"
        if data_split_type == "train":
            yaw_file, ped_file = "train_yaw_small.pkl", "train_ped_fix.pkl"
        elif data_split_type == "test":
            yaw_file, ped_file = "test_yaw_small.pkl", "test_ped_fix.pkl"
        else:
            raise ValueError(f"Invalid data_split_type: {data_split_type}")

        with open(f"{base_path}/{yaw_file}", "rb") as f:
            data = pickle.load(f)
        with open(f"{base_path}/{ped_file}", "rb") as f:
            data_ped = pickle.load(f)

        self.yaw_list = data[1]
        self.ped_list = data_ped[1]
        self.init_idx = data[0][0]

        self.total_size = math.ceil(len(self.frame_ids_range) / self.num_replicas) * self.num_replicas
        self.num_samples = self.total_size // self.num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator:
        g = torch.Generator()
        g.manual_seed(self.epoch + self.rank)

        indices_new = []
        for idx in tqdm(self.frame_ids_range, disable=(self.rank != 0)):
            thres_rate = random.random()
            ang_yaw = self.yaw_list[idx - self.init_idx] % (2 * math.pi)
            if ang_yaw > math.pi:
                ang_yaw -= 2 * math.pi

            while abs(ang_yaw) > 2.0:
                idx = random.choice(self.frame_ids_range)
                ang_yaw = self.yaw_list[idx - self.init_idx] % (2 * math.pi)
                if ang_yaw > math.pi:
                    ang_yaw -= 2 * math.pi

            if thres_rate < 0.5:
                while not (0.4 < abs(ang_yaw) < 2.0):
                    idx = random.choice(self.frame_ids_range)
                    ang_yaw = self.yaw_list[idx - self.init_idx] % (2 * math.pi)
                    if ang_yaw > math.pi:
                        ang_yaw -= 2 * math.pi

            indices_new.append(idx)

        if self.shuffle:
            indices_new = [indices_new[i] for i in torch.randperm(len(indices_new), generator=g)]

        if len(indices_new) < self.total_size:
            indices_new += indices_new[:self.total_size - len(indices_new)]
        assert len(indices_new) == self.total_size

        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(indices_new[start:end])

    def __len__(self) -> int:
        return self.num_samples
