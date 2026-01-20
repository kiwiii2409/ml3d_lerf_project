# Mod1: i tried to follow the other dataloaders as best as possible
import torch
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import Tuple
import os
import json
from lerf.data.utils.feature_dataloader import FeatureDataloader

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except ImportError:
    print("Segement anything not installed")


class SAMDataloader(FeatureDataloader):
    sam_model_type = "vit_h"
    points_per_side = 32
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        image_list: torch.Tensor,
        cache_path: Path,
    ):
        super().__init__(cfg, device, image_list, cache_path)

    def create(self, image_list):
        os.makedirs(self.cache_path, exist_ok=True)

        sam_ckpt = self.cfg["sam_ckpt_path"]
        sam = sam_model_registry[self.sam_model_type](checkpoint=(sam_ckpt))
        sam.to(device=self.device)

        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self.points_per_side,
            min_mask_region_area=100, # remove small disconnected patches (smaller than 100)
        )

        for idx, img_tensor in enumerate(tqdm(image_list, desc="sam")):
            img_arr = img_tensor.permute(1, 2, 0).cpu().numpy() # push channel dim to end chw to hwc
            img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)

            masks_result = mask_generator.generate(img_arr) # requires hwc uint 8
            if len(masks_result) > 0:
                masks_stack = np.stack([m['segmentation'] for m in masks_result])
                union_mask = masks_stack.any(axis=0) # (h,w) mask
            else:
                union_mask = np.zeros(img_arr.shape[:2], dtype=bool)

            save_path = self.cache_path / f"mask_{idx:05d}.npz"
            np.savez_compressed(save_path, masks=union_mask)
        
        # avoid OOM
        del mask_generator
        del sam
        torch.cuda.empty_cache()

    def get_masks_for_image(self, image_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Docstring for get_masks_for_image
        :return: shape (h,w) mask
        """
        file_path = self.cache_path / f"mask_{image_idx:05d}.npz"
        data = np.load(file_path)
        masks = torch.from_numpy(data['masks']) 
        return masks
    
    def load(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        if not cache_info_path.exists():
            raise FileNotFoundError 

        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        
        if cfg != self.cfg: # compare configs
            if self.cache_path.exists(): # if changed wipe folder and trigger create + save in parent
                shutil.rmtree(self.cache_path)
            raise ValueError("Config mismatch")

        if not self.cache_path.exists(): # check whether mask folder exists
             raise FileNotFoundError
        # if no error the files should exist

    def save(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))
    
    def __call__(self, img_points):
        return None