#Cell 1:
from google.colab import drive
drive.mount('/content/drive')

#Cell 2:
import csv
import json
from collections import defaultdict


def read_csv(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

def process_data(rectangle_data, point_data):
    processed_data = defaultdict(lambda: {'bboxes': [], 'keypoints': []})

    for rectangle_entry in rectangle_data:
        image_name = rectangle_entry[5]
        bbox_info = {
            'label_name': rectangle_entry[0],
            'bbox_x': rectangle_entry[1],
            'bbox_y': rectangle_entry[2],
            'bbox_width': rectangle_entry[3],
            'bbox_height': rectangle_entry[4]
        }
        processed_data[image_name]['bboxes'].append(bbox_info)

    for point_entry in point_data:
        image_name = point_entry[3]
        keypoint_info = {
            'label_name': point_entry[0],
            'x': point_entry[1],
            'y': point_entry[2]
        }
        processed_data[image_name]['keypoints'].append(keypoint_info)

    return processed_data

def generate_json_files(processed_data):
    for image_name, data in processed_data.items():
        with open(f'/content/drive/MyDrive/TennisMLProject/{image_name.split(".")[0]}.json', 'w') as jsonfile:
            json.dump(data, jsonfile, indent=4)

#Cell 3:
rectangle_data = read_csv('Rectangle.csv')
point_data = read_csv('Point.csv')
processed_data = process_data(rectangle_data, point_data)
generate_json_files(processed_data)

#Cell 4:
import os, cv2, numpy as np, matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import albumentations as A # Library for augmentations

import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
#from coco_eval import CocoEvaluator
#from coco_utils import get_coco_api_from_dataset

import copy
import io
from contextlib import redirect_stdout

import numpy as np
import pycocotools.mask as mask_util
import torch
#import utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

import os

import torch
import torch.utils.data
import torchvision

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
from torch import nn, Tensor
from torchvision import ops
from torchvision.transforms import functional as F, InterpolationMode, transforms as T

#Cell #5:
def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target


class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class ToDtype(nn.Module):
    def __init__(self, dtype: torch.dtype, scale: bool = False) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if not self.scale:
            return image.to(dtype=self.dtype), target
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class RandomIoUCrop(nn.Module):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
    ):
        super().__init__()
        # Configuration similar to https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L89-L174
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_h, orig_w = F.get_dimensions(image)

        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return image, target

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                boxes = target["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(
                    boxes, torch.tensor([[left, top, right, bottom]], dtype=boxes.dtype, device=boxes.device)
                )
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and perform cropping
                target["boxes"] = boxes
                target["labels"] = target["labels"][is_within_crop_area]
                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)
                image = F.crop(image, top, left, new_h, new_w)

                return image, target


class RandomZoomOut(nn.Module):
    def __init__(
        self, fill: Optional[List[float]] = None, side_range: Tuple[float, float] = (1.0, 4.0), p: float = 0.5
    ):
        super().__init__()
        if fill is None:
            fill = [0.0, 0.0, 0.0]
        self.fill = fill
        self.side_range = side_range
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(f"Invalid canvas side range provided {side_range}.")
        self.p = p

    @torch.jit.unused
    def _get_fill_value(self, is_pil):
        # type: (bool) -> int
        # We fake the type to make it work on JIT
        return tuple(int(x) for x in self.fill) if is_pil else 0

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        if torch.rand(1) >= self.p:
            return image, target

        _, orig_h, orig_w = F.get_dimensions(image)

        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)

        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)

        if torch.jit.is_scripting():
            fill = 0
        else:
            fill = self._get_fill_value(F._is_pil_image(image))

        image = F.pad(image, [left, top, right, bottom], fill=fill)
        if isinstance(image, torch.Tensor):
            # PyTorch's pad supports only integers on fill. So we need to overwrite the colour
            v = torch.tensor(self.fill, device=image.device, dtype=image.dtype).view(-1, 1, 1)
            image[..., :top, :] = image[..., :, :left] = image[..., (top + orig_h) :, :] = image[
                ..., :, (left + orig_w) :
            ] = v

        if target is not None:
            target["boxes"][:, 0::2] += left
            target["boxes"][:, 1::2] += top

        return image, target


class RandomPhotometricDistort(nn.Module):
    def __init__(
        self,
        contrast: Tuple[float, float] = (0.5, 1.5),
        saturation: Tuple[float, float] = (0.5, 1.5),
        hue: Tuple[float, float] = (-0.05, 0.05),
        brightness: Tuple[float, float] = (0.875, 1.125),
        p: float = 0.5,
    ):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < 0.5
        if contrast_before:
            if r[2] < self.p:
                image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if not contrast_before:
            if r[5] < self.p:
                image = self._contrast(image)

        if r[6] < self.p:
            channels, _, _ = F.get_dimensions(image)
            permutation = torch.randperm(channels)

            is_pil = F._is_pil_image(image)
            if is_pil:
                image = F.pil_to_tensor(image)
                image = F.convert_image_dtype(image)
            image = image[..., permutation, :, :]
            if is_pil:
                image = F.to_pil_image(image)

        return image, target


class ScaleJitter(nn.Module):
    """Randomly resizes the image and its bounding boxes  within the specified scale range.
    The class implements the Scale Jitter augmentation as described in the paper
    `"Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" <https://arxiv.org/abs/2012.07177>`_.

    Args:
        target_size (tuple of ints): The target size for the transform provided in (height, weight) format.
        scale_range (tuple of ints): scaling factor interval, e.g (a, b), then scale is randomly sampled from the
            range a <= scale <= b.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.1, 2.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias=True,
    ):
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_height, orig_width = F.get_dimensions(image)

        scale = self.scale_range[0] + torch.rand(1) * (self.scale_range[1] - self.scale_range[0])
        r = min(self.target_size[1] / orig_height, self.target_size[0] / orig_width) * scale
        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation, antialias=self.antialias)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"],
                    [new_height, new_width],
                    interpolation=InterpolationMode.NEAREST,
                    antialias=self.antialias,
                )

        return image, target


class FixedSizeCrop(nn.Module):
    def __init__(self, size, fill=0, padding_mode="constant"):
        super().__init__()
        size = tuple(T._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.crop_height = size[0]
        self.crop_width = size[1]
        self.fill = fill  # TODO: Fill is currently respected only on PIL. Apply tensor patch.
        self.padding_mode = padding_mode

    def _pad(self, img, target, padding):
        # Taken from the functional_tensor.py pad
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        elif len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        else:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

        padding = [pad_left, pad_top, pad_right, pad_bottom]
        img = F.pad(img, padding, self.fill, self.padding_mode)
        if target is not None:
            target["boxes"][:, 0::2] += pad_left
            target["boxes"][:, 1::2] += pad_top
            if "masks" in target:
                target["masks"] = F.pad(target["masks"], padding, 0, "constant")

        return img, target

    def _crop(self, img, target, top, left, height, width):
        img = F.crop(img, top, left, height, width)
        if target is not None:
            boxes = target["boxes"]
            boxes[:, 0::2] -= left
            boxes[:, 1::2] -= top
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)

            is_valid = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])

            target["boxes"] = boxes[is_valid]
            target["labels"] = target["labels"][is_valid]
            if "masks" in target:
                target["masks"] = F.crop(target["masks"][is_valid], top, left, height, width)

        return img, target

    def forward(self, img, target=None):
        _, height, width = F.get_dimensions(img)
        new_height = min(height, self.crop_height)
        new_width = min(width, self.crop_width)

        if new_height != height or new_width != width:
            offset_height = max(height - self.crop_height, 0)
            offset_width = max(width - self.crop_width, 0)

            r = torch.rand(1)
            top = int(offset_height * r)
            left = int(offset_width * r)

            img, target = self._crop(img, target, top, left, new_height, new_width)

        pad_bottom = max(self.crop_height - new_height, 0)
        pad_right = max(self.crop_width - new_width, 0)
        if pad_bottom != 0 or pad_right != 0:
            img, target = self._pad(img, target, [0, 0, pad_right, pad_bottom])

        return img, target


class RandomShortestSize(nn.Module):
    def __init__(
        self,
        min_size: Union[List[int], Tuple[int], int],
        max_size: int,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.min_size = [min_size] if isinstance(min_size, int) else list(min_size)
        self.max_size = max_size
        self.interpolation = interpolation

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        _, orig_height, orig_width = F.get_dimensions(image)

        min_size = self.min_size[torch.randint(len(self.min_size), (1,)).item()]
        r = min(min_size / min(orig_height, orig_width), self.max_size / max(orig_height, orig_width))

        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"], [new_height, new_width], interpolation=InterpolationMode.NEAREST
                )

        return image, target


def _copy_paste(
    image: torch.Tensor,
    target: Dict[str, Tensor],
    paste_image: torch.Tensor,
    paste_target: Dict[str, Tensor],
    blending: bool = True,
    resize_interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
) -> Tuple[torch.Tensor, Dict[str, Tensor]]:

    # Random paste targets selection:
    num_masks = len(paste_target["masks"])

    if num_masks < 1:
        # Such degerante case with num_masks=0 can happen with LSJ
        # Let's just return (image, target)
        return image, target

    # We have to please torch script by explicitly specifying dtype as torch.long
    random_selection = torch.randint(0, num_masks, (num_masks,), device=paste_image.device)
    random_selection = torch.unique(random_selection).to(torch.long)

    paste_masks = paste_target["masks"][random_selection]
    paste_boxes = paste_target["boxes"][random_selection]
    paste_labels = paste_target["labels"][random_selection]

    masks = target["masks"]

    # We resize source and paste data if they have different sizes
    # This is something we introduced here as originally the algorithm works
    # on equal-sized data (for example, coming from LSJ data augmentations)
    size1 = image.shape[-2:]
    size2 = paste_image.shape[-2:]
    if size1 != size2:
        paste_image = F.resize(paste_image, size1, interpolation=resize_interpolation)
        paste_masks = F.resize(paste_masks, size1, interpolation=F.InterpolationMode.NEAREST)
        # resize bboxes:
        ratios = torch.tensor((size1[1] / size2[1], size1[0] / size2[0]), device=paste_boxes.device)
        paste_boxes = paste_boxes.view(-1, 2, 2).mul(ratios).view(paste_boxes.shape)

    paste_alpha_mask = paste_masks.sum(dim=0) > 0

    if blending:
        paste_alpha_mask = F.gaussian_blur(
            paste_alpha_mask.unsqueeze(0),
            kernel_size=(5, 5),
            sigma=[
                2.0,
            ],
        )

    # Copy-paste images:
    image = (image * (~paste_alpha_mask)) + (paste_image * paste_alpha_mask)

    # Copy-paste masks:
    masks = masks * (~paste_alpha_mask)
    non_all_zero_masks = masks.sum((-1, -2)) > 0
    masks = masks[non_all_zero_masks]

    # Do a shallow copy of the target dict
    out_target = {k: v for k, v in target.items()}

    out_target["masks"] = torch.cat([masks, paste_masks])

    # Copy-paste boxes and labels
    boxes = ops.masks_to_boxes(masks)
    out_target["boxes"] = torch.cat([boxes, paste_boxes])

    labels = target["labels"][non_all_zero_masks]
    out_target["labels"] = torch.cat([labels, paste_labels])

    # Update additional optional keys: area and iscrowd if exist
    if "area" in target:
        out_target["area"] = out_target["masks"].sum((-1, -2)).to(torch.float32)

    if "iscrowd" in target and "iscrowd" in paste_target:
        # target['iscrowd'] size can be differ from mask size (non_all_zero_masks)
        # For example, if previous transforms geometrically modifies masks/boxes/labels but
        # does not update "iscrowd"
        if len(target["iscrowd"]) == len(non_all_zero_masks):
            iscrowd = target["iscrowd"][non_all_zero_masks]
            paste_iscrowd = paste_target["iscrowd"][random_selection]
            out_target["iscrowd"] = torch.cat([iscrowd, paste_iscrowd])

    # Check for degenerated boxes and remove them
    boxes = out_target["boxes"]
    degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
    if degenerate_boxes.any():
        valid_targets = ~degenerate_boxes.any(dim=1)

        out_target["boxes"] = boxes[valid_targets]
        out_target["masks"] = out_target["masks"][valid_targets]
        out_target["labels"] = out_target["labels"][valid_targets]

        if "area" in out_target:
            out_target["area"] = out_target["area"][valid_targets]
        if "iscrowd" in out_target and len(out_target["iscrowd"]) == len(valid_targets):
            out_target["iscrowd"] = out_target["iscrowd"][valid_targets]

    return image, out_target


class SimpleCopyPaste(torch.nn.Module):
    def __init__(self, blending=True, resize_interpolation=F.InterpolationMode.BILINEAR):
        super().__init__()
        self.resize_interpolation = resize_interpolation
        self.blending = blending

    def forward(
        self, images: List[torch.Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Tensor]]]:
        torch._assert(
            isinstance(images, (list, tuple)) and all([isinstance(v, torch.Tensor) for v in images]),
            "images should be a list of tensors",
        )
        torch._assert(
            isinstance(targets, (list, tuple)) and len(images) == len(targets),
            "targets should be a list of the same size as images",
        )
        for target in targets:
            # Can not check for instance type dict with inside torch.jit.script
            # torch._assert(isinstance(target, dict), "targets item should be a dict")
            for k in ["masks", "boxes", "labels"]:
                torch._assert(k in target, f"Key {k} should be present in targets")
                torch._assert(isinstance(target[k], torch.Tensor), f"Value for the key {k} should be a tensor")

        # images = [t1, t2, ..., tN]
        # Let's define paste_images as shifted list of input images
        # paste_images = [t2, t3, ..., tN, t1]
        # FYI: in TF they mix data on the dataset level
        images_rolled = images[-1:] + images[:-1]
        targets_rolled = targets[-1:] + targets[:-1]

        output_images: List[torch.Tensor] = []
        output_targets: List[Dict[str, Tensor]] = []

        for image, target, paste_image, paste_target in zip(images, targets, images_rolled, targets_rolled):
            output_image, output_data = _copy_paste(
                image,
                target,
                paste_image,
                paste_target,
                blending=self.blending,
                resize_interpolation=self.resize_interpolation,
            )
            output_images.append(output_image)
            output_targets.append(output_data)

        return output_images, output_targets

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(blending={self.blending}, resize_interpolation={self.resize_interpolation})"
        return s

#
#
#

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"]
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    # FIXME: This is... awful?
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_coco(root, image_set, transforms, mode="instances", use_v2=False, with_masks=False):
    anno_file_template = "{}_{}2017.json"
    PATHS = {
        "train": ("train2017", os.path.join("annotations", anno_file_template.format(mode, "train"))),
        "val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    if use_v2:
        from torchvision.datasets import wrap_dataset_for_transforms_v2

        dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)
        target_keys = ["boxes", "labels", "image_id"]
        if with_masks:
            target_keys += ["masks"]
        dataset = wrap_dataset_for_transforms_v2(dataset, target_keys=target_keys)
    else:
        # TODO: handle with_masks for V1?
        t = [ConvertCocoPolysToMask()]
        if transforms is not None:
            t.append(transforms)
        transforms = T.Compose(t)

        dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset

#
#
#

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

###
###

class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead  got {type(iou_types)}")
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            coco_eval = COCOeval(coco_gt, iouType=iou_type)
            coco_eval.params.kpt_oks_sigmas = np.array([.9, .9, .9, .9, .9]) / 10.0
            self.coco_eval[iou_type] = coco_eval

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate_singlearg(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

###
###
###

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def evaluate_singlearg(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))

#@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

#Cell 6:
#!curl -o coco_eval.py https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py
#!curl -o coco_utils.py https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py
#!curl -o engine.py https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py
#!curl -o group_by_aspect_ratio.py https://raw.githubusercontent.com/pytorch/vision/main/references/detection/group_by_aspect_ratio.py
#!curl -o presets.py https://raw.githubusercontent.com/pytorch/vision/main/references/detection/presets.py
#!curl -o train.py https://raw.githubusercontent.com/pytorch/vision/main/references/detection/train.py
#!curl -o transforms.py https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py
!curl -o utils.py https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py

#Cell 7:
#import transforms, utils, engine, train
import utils
#from utils import collate_fn
#from engine import train_one_epoch, evaluate

#Cell 8:
#self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
#coco_eval = COCOeval(coco_gt, iouType=iou_type)
#coco_eval.params.kpt_oks_sigmas = np.array([.5, .5, .5, .5, .5]) / 10.0
#self.coco_eval[iou_type] = coco_eval

#Cell 9:
def train_transform():
    return A.Compose([
        A.Sequential([
            A.RandomRotate90(p=1), # Random rotation of an image by 90 degrees zero or more times
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more at https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )

#Cell 10:
class ClassDataset(Dataset):
   def __init__(self, root, indices, transform=None, demo=False):
       self.root = root
       #print(root)
       self.transform = transform
       self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
       self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
       self.imgs_files = [self.imgs_files[i] for i in indices]
       #print(len(self.imgs_files))
       self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))
       self.annotations_files = [self.annotations_files[i] for i in indices]


   def __getitem__(self, idx):
       img_path = os.path.join(self.root, "images", self.imgs_files[idx])
       annotations_path = os.path.join(self.root, "annotations", self.annotations_files[idx])


       img_original = cv2.imread(img_path)
       img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)


       with open(annotations_path) as f:
           data = json.load(f)
           bboxes_original = data['bboxes'][0]
           x0, y0 = int(bboxes_original['bbox_x']), int(bboxes_original['bbox_y'])
           x1, y1 = x0 + int(bboxes_original['bbox_width']), y0 + int(bboxes_original['bbox_height'])
           bboxes_original = [[x0, y0, x1, y1]]

           #List will be in order of Center, Top, Bottom, Left, Right
           keypoints_list_of_lists=[
               [0, 0, 1], #Center
               [0, 0, 1], #Top
               [0, 0, 1], #Bottom
               [0, 0, 1], #Left
               [0, 0, 1]  #Right
           ]

           keypoints_original = data['keypoints']
           for keypoint in keypoints_original:
             #print(keypoint["label_name"])
             keypoint_name=keypoint["label_name"]
             x, y = int(keypoint["x"]), int(keypoint["y"])
             if keypoint_name=='Center':
               keypoints_list_of_lists[0] = [x, y, 1]
             elif keypoint_name == 'Top':
               keypoints_list_of_lists[1] = [x, y, 1]
             elif keypoint_name == 'Bottom':
               keypoints_list_of_lists[2] = [x, y, 1]
             elif keypoint_name == 'Left':
               keypoints_list_of_lists[3] = [x, y, 1]
             elif keypoint_name == 'Right':
               keypoints_list_of_lists[4] = [x, y, 1]
           #print(keypoints_list_of_lists)
           keypoints_original=[keypoints_list_of_lists]

           # All objects are glue tubes
           bboxes_labels_original = ['Glue tube' for _ in bboxes_original]


       if self.transform:
           # Converting keypoints from [x,y,visibility]-format to [x, y]-format + Flattening nested list of keypoints
           # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
           # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format
           # Then we need to convert it to the following list:
           # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
           keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]


           # Apply augmentations
           transformed = self.transform(image=img_original, bboxes=bboxes_original, bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened)
           img = transformed['image']
           bboxes = transformed['bboxes']


           # Unflattening list transformed['keypoints']
           # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
           # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2], where each keypoint is in [x, y]-format
           # Then we need to convert it to the following list:
           # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
           keypoints_transformed_unflattened = np.reshape(np.array(transformed['keypoints']), (-1,5,2)).tolist()


           # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
           keypoints = []
           for o_idx, obj in enumerate(keypoints_transformed_unflattened): # Iterating over objects
               obj_keypoints = []
               for k_idx, kp in enumerate(obj): # Iterating over keypoints in each object
                   # kp - coordinates of keypoint
                   # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
                   obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
               keypoints.append(obj_keypoints)


       else:
           img, bboxes, keypoints = img_original, bboxes_original, keypoints_original


       # Convert everything into a torch tensor
       bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
       target = {}
       target["boxes"] = bboxes
       target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64) # all objects are glue tubes
       target["image_id"] = idx # torch.tensor([idx])
       target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
       target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
       target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
       img = F.to_tensor(img)


       bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
       target_original = {}
       target_original["boxes"] = bboxes_original
       target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original], dtype=torch.int64) # all objects are glue tubes
       target_original["image_id"] = idx #torch.tensor([idx])
       target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0])
       target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
       target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)
       img_original = F.to_tensor(img_original)


       if self.demo:
           return img, target, img_original, target_original
       else:
           return img, target


   def __len__(self):
       return len(self.imgs_files)

#Cell 11:
from torch.utils.data import random_split
import pickle
import random

KEYPOINTS_FOLDER_TRAIN = '/content/drive/MyDrive/TennisMLProject'
#dataset = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform(), demo=False)

file_numbers = list(range(len(os.listdir(os.path.join(KEYPOINTS_FOLDER_TRAIN, "images")))))
train_indices = random.sample(file_numbers, k=int(len(file_numbers) * 0.7))
remaining_after_seventy = [num for num in file_numbers if num not in train_indices]
# Select 50% of the remaining numbers at random
val_indices = random.sample(remaining_after_seventy, k=int(len(remaining_after_seventy) * 0.5))
test_indices = [num for num in remaining_after_seventy if num not in val_indices]
# Define the sizes of your train, validation, and test sets
#train_size = int(0.7 * len(dataset))  # 70% of data for training
#val_size = int(0.15 * len(dataset))   # 15% of data for validation
#test_size = len(dataset) - train_size - val_size  # Remaining data for testing

# Use random_split to split the dataset
#train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

#with open('train_set.pickle', 'wb') as handle:
    #pickle.dump(train_indices, handle)
#with open('val_set.pickle', 'wb') as handle:
   #pickle.dump(val_indices, handle)
#with open('test_set.pickle', 'wb') as handle:
    #pickle.dump(test_indices, handle)
with open('train_set.pickle', 'rb') as handle:
    train_indices=pickle.load(handle)
with open('val_set.pickle', 'rb') as handle:
    val_indices=pickle.load(handle)
with open('test_set.pickle', 'rb') as handle:
    test_indices=pickle.load(handle)
traindataset = ClassDataset(KEYPOINTS_FOLDER_TRAIN, train_indices, transform=train_transform(), demo=False)
valdataset = ClassDataset(KEYPOINTS_FOLDER_TRAIN, val_indices, transform=train_transform(), demo=False)
testdataset = ClassDataset(KEYPOINTS_FOLDER_TRAIN, test_indices, transform=train_transform(), demo=False)
# Optionally, you can create DataLoader for each set
train_loader = DataLoader(traindataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(valdataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(testdataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

#iterator = iter(train_loader)
#batch = next(iterator)

#print("Original targets:\n", batch[3], "\n\n")
#print("Transformed targets:\n", batch[1])

#Cell 12:
#keypoints_classes_ids2names = {0: 'Center', 1: 'Top' , 2: 'Bottom' , 3: 'Left' , 4: 'Right'}

#def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    #fontsize = 18

   #for bbox in bboxes:
        #start_point = (bbox[0], bbox[1])
        #end_point = (bbox[2], bbox[3])
        #image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)

   # for kps in keypoints:
        #for idx, kp in enumerate(kps):
           # image = cv2.circle(image.copy(), tuple(kp), 5, (255,0,0), 10)
            #image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

    #if image_original is None and keypoints_original is None:
        #plt.figure(figsize=(40,40))
        #plt.imshow(image)

    #else:
        #for bbox in bboxes_original:
            #start_point = (bbox[0], bbox[1])
            #end_point = (bbox[2], bbox[3])
            #image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)

        #for kps in keypoints_original:
            #for idx, kp in enumerate(kps):
               #image_original = cv2.circle(image_original, tuple(kp), 5, (255,0,0), 10)
                #image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)

        #f, ax = plt.subplots(1, 2, figsize=(40, 20))

        #ax[0].imshow(image_original)
        #ax[0].set_title('Original image', fontsize=fontsize)

        #ax[1].imshow(image)
        #ax[1].set_title('Transformed image', fontsize=fontsize)

#image = (batch[0][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
#bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

#keypoints = []
#for kps in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
    #keypoints.append([kp[:2] for kp in kps])

#image_original = (batch[2][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
#bboxes_original = batch[3][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

#keypoints_original = []
#for kps in batch[3][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
    #keypoints_original.append([kp[:2] for kp in kps])

#visualize(image, bboxes, keypoints, image_original, bboxes_original, keypoints_original)

#Cell 13:
def get_model(num_keypoints, weights_path=None):

    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)

    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    return model

#Cell 14:
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
KEYPOINTS_FOLDER_TRAIN = '/content/drive/MyDrive/TennisMLProject'
#KEYPOINTS_FOLDER_TEST = '/content/drive/MyDrive/TennisMLProject'

#dataset_train = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform(), demo=False)
#dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

#data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, collate_fn=collate_fn)
#data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = get_model(num_keypoints = 5)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
num_epochs = 100

best_val = 0

for epoch in range(num_epochs):
    model.train()
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1000)
    lr_scheduler.step()
    #print(evaluate.__module__)
    model.eval()
    val_stats = evaluate(model, val_loader, device)
    val_loss = val_stats.coco_eval["keypoints"].stats[0]
    print("Validation Loss: "+str(val_loss))
    if val_loss >= best_val:
      # Save model weights after training
      torch.save(model.state_dict(), '/content/drive/MyDrive/BestTennisModel.pth')

#Cell 15:
model = get_model(num_keypoints = 5)
model.load_state_dict(torch.load('/content/drive/MyDrive/BestTennisModel.pth'))

# Ensure the model is in evaluation mode
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

test_stats = evaluate(model, test_loader, device)

#Cell 16:
# Assuming test_loader is already defined and properly initialized as shown in previous sections
iterator = iter(test_loader)

# Load a batch of images and targets from the test dataset
images, targets = next(iterator)

# Move the images to the appropriate device (GPU or CPU)
images = list(image.to(device) for image in images)

# Disable gradient calculation to speed up the process and reduce memory usage
# because gradients are not needed for model evaluation
with torch.no_grad():
    # Ensure the model is on the correct device and set it to evaluation mode
    model.to(device)
    model.eval()

    # Generate model predictions for the given batch of images
    output = model(images)

# Print the predictions; you might want to format or process them further depending on your needs
print("Predictions: \n", output)

#Cell 17:
import numpy as np
import torchvision
import matplotlib.pyplot as plt

# Function to visualize an image with bounding boxes and keypoints
def visualize(image, bboxes, keypoints):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for bbox, kps in zip(bboxes, keypoints):
        plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                          fill=False, edgecolor='red', linewidth=2))
        for kp in kps:
            plt.plot(kp[0], kp[1], 'ro', markersize=5)
    plt.axis('off')
    plt.show()

okscollection = []
iterator = iter(test_loader)
model.to(device)
model.eval()
i = 0

while True:
    print(i)
    i += 1
    try:
        # Load a batch of images and targets from the test dataset
        images, targets = next(iterator)

        # Move the images to the appropriate device (GPU or CPU)
        images = list(image.to(device) for image in images)

        # Disable gradient calculation
        with torch.no_grad():
            # Generate model predictions for the given batch of images
            output = model(images)

        if len(output[0]["keypoints"]) > 0:
            a = targets[0]["keypoints"].tolist()[0]
            b = output[0]["keypoints"].tolist()[0]
            a = [kp[:2] for kp in a]
            b = [kp[:2] for kp in b]
            okscollection.append(compute_coco_oks(b, a))
        else:
            okscollection.append(0)

        # Visualization logic
        image = (images[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0]
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()
        keypoints = [output[0]['keypoints'][idx].detach().cpu().numpy()[:, :2].astype(int).tolist() for idx in post_nms_idxs]
        bboxes = [output[0]['boxes'][idx].detach().cpu().numpy().astype(int).tolist() for idx in post_nms_idxs]

        # Visualize the processed image with keypoints and bounding boxes
        visualize(image, bboxes, keypoints)

    except StopIteration:
        break

plt.hist(okscollection)
plt.show()
print(np.mean(okscollection))
print(np.var(okscollection))

#Cell 18:
def compute_coco_oks(predicted_keypoints, ground_truth_keypoints, sigma=0.1):
    """
    Compute the object-keypoint similarity score in COCO OKS (Object Keypoint Similarity) way.

    Parameters:
    - predicted_keypoints: List of predicted keypoints coordinates [(x1, y1), (x2, y2), ..., (xn, yn)].
    - ground_truth_keypoints: List of ground truth keypoints coordinates [(x1_gt, y1_gt), (x2_gt, y2_gt), ..., (xn_gt, yn_gt)].
    - sigma: Sigma value for Gaussian function.

    Returns:
    - similarity_score: The object-keypoint similarity score in COCO OKS way.
    """

    # Convert keypoints to NumPy arrays for easier manipulation
    pred_kp = np.array(predicted_keypoints)
    gt_kp = np.array(ground_truth_keypoints)
    #Calculate the distance between each predicted keypoint and its nearest ground truth keypoint
    distances = np.linalg.norm(pred_kp[:, None] - gt_kp, axis=-1)
    min_distances = np.min(distances, axis=1)
    area = (max([a[1] for a in ground_truth_keypoints]) - min([a[1] for a in ground_truth_keypoints]))*(max([a[0] for a in ground_truth_keypoints]) - min([a[0] for a in ground_truth_keypoints]))
    # Compute the similarity score using the COCO OKS formula

    similarity_score = np.exp(-min_distances ** 2 / (2 * area * sigma ** 2))

    # Average the scores
    similarity_score = np.mean(similarity_score)

    return similarity_score

#Cell 19:
model = get_model(num_keypoints = 5)
model.load_state_dict(torch.load('/content/drive/MyDrive/BestTennisModel.pth'))

# Ensure the model is in evaluation mode
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Assuming test_loader is already defined and properly initialized as shown in previous sections
iterator = iter(test_loader)

# Load a batch of images and targets from the test dataset
images, targets = next(iterator)

# Move the images to the appropriate device (GPU or CPU)
images = list(image.to(device) for image in images)

# Disable gradient calculation to speed up the process and reduce memory usage
# because gradients are not needed for model evaluation
with torch.no_grad():
    # Ensure the model is on the correct device and set it to evaluation mode
    model.to(device)
    model.eval()

    # Generate model predictions for the given batch of images
    output = model(images)

# Print the predictions; you might want to format or process them further depending on your needs
print("Predictions: \n", output)
print(targets)
print(targets[0]["keypoints"].tolist())
print(output[0]["keypoints"].tolist())

#Cell 20:
okscollection= []
iterator = iter(test_loader)
model.to(device)
model.eval()
i=0
while True:
  print(i)
  i+=1
  try:

    # Load a batch of images and targets from the test dataset
    images, targets = next(iterator)

    # Move the images to the appropriate device (GPU or CPU)
    images = list(image.to(device) for image in images)

    # Disable gradient calculation to speed up the process and reduce memory usage
    # because gradients are not needed for model evaluation
    with torch.no_grad():
        # Generate model predictions for the given batch of images
        output = model(images)
    if len(output[0]["keypoints"]) > 0:
      a=targets[0]["keypoints"].tolist()[0]
      b=output[0]["keypoints"].tolist()[0]
      a = [kp[:2] for kp in a]
      b = [kp[:2] for kp in b]
      okscollection+=[compute_coco_oks(b, a)]
    else:
      okscollection+=[0]
  except StopIteration:
    break
plt.hist(okscollection)
plt.show()
print(np.mean(okscollection))
print(np.var(okscollection))
