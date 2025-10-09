"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_map_with_key(fn: Callable, tree: dict, keys: Sequence = ()) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map_with_key(fn, v, (*keys, k)) if isinstance(v, dict) else fn((*keys, k), v) for k, v in tree.items()
    }


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        #print("self.pad_token_id", self.pad_token_id)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )

@dataclass
class PaddedCollatorForActionPrediction_Nav:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_wrist" in instances[0]:
                pixel_values_wrist = [instance["pixel_values_wrist"] for instance in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_wrist)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        # Stack goal_pose
        goal_pose = [torch.from_numpy(np.copy(instance["goal_pose"])) for instance in instances]
        goal_pose = torch.stack(goal_pose)

        # Stack obj_pose
        obj_pose_norm = [torch.from_numpy(np.copy(instance["obj_pose_norm"])) for instance in instances]
        obj_pose_norm = torch.stack(obj_pose_norm)

        # Stack cur_image
        cur_image_crop = [torch.from_numpy(np.copy(instance["cur_image_crop"])) for instance in instances]
        cur_image_crop = torch.stack(cur_image_crop)

        # Stack cur_image
        cur_image = [torch.from_numpy(np.copy(instance["cur_image"])) for instance in instances]
        cur_image = torch.stack(cur_image)

        # Stack goal_image_crop
        goal_image_crop = [torch.from_numpy(np.copy(instance["goal_image_crop"])) for instance in instances]
        goal_image_crop = torch.stack(goal_image_crop)

        # Stack goal_image_8
        goal_image_8 = [torch.from_numpy(np.copy(instance["goal_image_8"])) for instance in instances]
        goal_image_8 = torch.stack(goal_image_8)

        # Stack temp_dist
        temp_dist = [torch.from_numpy(np.copy(instance["temp_dist"])) for instance in instances]
        temp_dist = torch.stack(temp_dist)
        
        # Stack proprio
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.squeeze(np.stack(proprio)))
        else:
            proprio = None

        output = dict(
            pixel_values=pixel_values,
            proprio=proprio,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            goal_pose=goal_pose,
            obj_pose_norm=obj_pose_norm,
            cur_image_crop=cur_image_crop,
            cur_image=cur_image,
            goal_image_crop=goal_image_crop,
            goal_image_8=goal_image_8,       
            temp_dist=temp_dist,     
            img_PIL=[instance["img_PIL"] for instance in instances],
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

@dataclass
class PaddedCollatorForActionPrediction_Nav_MMN:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32
    num_img: int = 1

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        #input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
        #input_ids = [ids[:self.model_max_length] for ids in input_ids]
        #labels = [lbl[:self.model_max_length] for lbl in labels]
        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        """
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)
        """
        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_wrist" in instances[0] and self.num_img > 1:
                if self.num_img == 2:
                    pixel_values_wrist = [instance["pixel_values_wrist"] for instance in instances]
                    pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_wrist)), dim=1)
                    #pixel_values_curmap = [instance["pixel_values_curmap"] for instance in instances]
                    #pixel_values_goalmap = [instance["pixel_values_goalmap"] for instance in instances]
                    #pixel_values = torch.cat((torch.stack(pixel_values_curmap), torch.stack(pixel_values_goalmap)), dim=1)                      
                else:
                    pixel_values_wrist = [instance["pixel_values_wrist"] for instance in instances]
                    pixel_values_curmap = [instance["pixel_values_curmap"] for instance in instances]
                    pixel_values_goalmap = [instance["pixel_values_goalmap"] for instance in instances]
                    pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_wrist), torch.stack(pixel_values_curmap), torch.stack(pixel_values_goalmap)), dim=1)                
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)
        actions_nomad = [torch.from_numpy(np.copy(instance["actions_nomad"])) for instance in instances]
        actions_nomad = torch.stack(actions_nomad)

        # Stack goal_pose
        goal_pose = [torch.from_numpy(np.copy(instance["goal_pose"])) for instance in instances]
        goal_pose = torch.stack(goal_pose)

        # Stack obj_pose
        obj_pose_norm = [torch.from_numpy(np.copy(instance["obj_pose_norm"])) for instance in instances]
        obj_pose_norm = torch.stack(obj_pose_norm)

        # Stack cur_image
        cur_image_crop = [torch.from_numpy(np.copy(instance["cur_image_crop"])) for instance in instances]
        cur_image_crop = torch.stack(cur_image_crop)

        # Stack cur_image
        cur_image = [torch.from_numpy(np.copy(instance["cur_image"])) for instance in instances]
        cur_image = torch.stack(cur_image)

        # Stack goal_image_crop
        goal_image_crop = [torch.from_numpy(np.copy(instance["goal_image_crop"])) for instance in instances]
        goal_image_crop = torch.stack(goal_image_crop)

        # Stack goal_image_8
        goal_image_8 = [torch.from_numpy(np.copy(instance["goal_image_8"])) for instance in instances]
        goal_image_8 = torch.stack(goal_image_8)

        # Stack temp_dist
        temp_dist = [torch.from_numpy(np.copy(instance["temp_dist"])) for instance in instances]
        temp_dist = torch.stack(temp_dist)

        # Stack current map image
        cur_map_image = [torch.from_numpy(np.copy(instance["cur_map_image"])) for instance in instances]
        cur_map_image = torch.stack(cur_map_image)

        # Stack goal map image
        goal_map_image = [torch.from_numpy(np.copy(instance["goal_map_image"])) for instance in instances]
        goal_map_image = torch.stack(goal_map_image)
        
        # Stack proprio
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.squeeze(np.stack(proprio)))
        else:
            proprio = None
        #print(input_ids.size(), labels.size())

        output = dict(
            pixel_values=pixel_values,
            proprio=proprio,
            input_ids=input_ids,
            #attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            actions_nomad=actions_nomad,
            goal_pose=goal_pose,
            obj_pose_norm=obj_pose_norm,
            cur_image_crop=cur_image_crop,
            cur_image=cur_image,
            goal_image_crop=goal_image_crop,
            goal_image_8=goal_image_8,       
            temp_dist=temp_dist,     
            cur_map_image=cur_map_image,
            goal_map_image=goal_map_image,
            img_PIL=[instance["img_PIL"] for instance in instances],
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output

@dataclass
class PaddedCollatorForActionPrediction_Nav_lelan:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_wrist" in instances[0]:
                pixel_values_wrist = [instance["pixel_values_wrist"] for instance in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_wrist)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        # Stack goal_pose
        goal_pose = [torch.from_numpy(np.copy(instance["goal_pose"])) for instance in instances]
        goal_pose = torch.stack(goal_pose)

        # Stack obj_pose
        obj_pose_norm = [torch.from_numpy(np.copy(instance["obj_pose_norm"])) for instance in instances]
        obj_pose_norm = torch.stack(obj_pose_norm)
        
        # Stack proprio
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.squeeze(np.stack(proprio)))
        else:
            proprio = None

        output = dict(
            pixel_values=pixel_values,
            proprio=proprio,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            goal_pose=goal_pose,
            obj_pose_norm=obj_pose_norm,
            #cur_image_crop=cur_image_crop,
            #cur_image=cur_image,
            #goal_image_crop=goal_image_crop,
            #goal_image_8=goal_image_8,       
            #temp_dist=temp_dist,     
            img_PIL=[instance["img_PIL"] for instance in instances],
            inst=[instance["inst"] for instance in instances]
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output
        
@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None

        # For now, we only support Tokenizers with `padding_side = "right"` during training
        #   => Handle padding via RNN Utils => `pad_sequence`
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_wrist" in instances[0]:
                pixel_values_wrist = [instance["pixel_values_wrist"] for instance in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_wrist)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        # Stack proprio
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.squeeze(np.stack(proprio)))
        else:
            proprio = None

        output = dict(
            pixel_values=pixel_values,
            proprio=proprio,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output
