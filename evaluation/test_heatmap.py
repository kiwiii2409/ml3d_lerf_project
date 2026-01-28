import numpy as np 
from pathlib import Path
from typing import Optional

import torch
from test_base import BaseTest





class Heatmap_Test(BaseTest):
    def __init__(self, name:str = "Heatmap-SAM-IoU", ground_truth_path: Optional[Path] = None, relevancy_threshold: float = 0.5):
        self.relevancy_threshold = relevancy_threshold
        super().__init__(name, ground_truth_path)

        self.results = [] # store iou per image per prompt
        self.detailed_results = {}

    # TODO:(cahnge this to _get_ground_truth) to call sam on list of images and store in dictionary {filename: mask_dict}
    # keep sam masks separate (dont merge to object) and calcualte iou with all masks => find best fit as final iou value 
    # otherwise querying a single object in a scene with multiple objects will always lead to bad performance, as union_obj_mask is larger than single object
    def _load_sam_masks(self, image_idx):
        path = self.ground_truth_path / f"mask_{image_idx:05d}.npz"
        try:
            data = np.load(path)
            if 'masks' in data:
                return torch.from_numpy(data['masks'])
        except Exception:
            raise Exception(f"Invalid File Path to SAM Masks: {path}")
            
    def evaluate_image(self, image_idx: int, rendered_layers: dict, model, gt_meta: dict):
        sam_mask = self._load_sam_masks(image_idx) # also shape HxW but maybe different
        
        target_prompts = gt_meta["positives"]


        for prompt in target_prompts:
            promtp_idx = model.image_encoder.positives.index(prompt)
            relevancy_key = f"relevancy_{promtp_idx}"
            relevancy_map = rendered_layers[relevancy_key].squeeze() 
            pred_mask = (relevancy_map > self.relevancy_threshold).float() # shape HxW

            print(f"Image Idx: {image_idx} with prompt: {prompt}")
            print(f"Shape PredMask: {pred_mask.shape}")
            print(f"Shape SamMask: {sam_mask.shape}")
            if sam_mask.shape != pred_mask.shape: # not sure if needed
                raise Exception(f"Invalid shape! sam_mask: {sam_mask.shape} vs. pred_mask: {pred_mask.shape}")
            
            binary_sam_mask = (sam_mask > 0.5).float().to(pred_mask.device)

            intersection = (pred_mask * binary_sam_mask).sum()
            union = torch.max(pred_mask, binary_sam_mask).sum()            

            iou = intersection / (union + 1e-8)
            iou_val = iou.item()

            self.results.append(iou_val)

            if image_idx not in self.detailed_results:
                self.detailed_results[image_idx] = {}
            self.detailed_results[image_idx][prompt] = iou_val
        
    def summarize(self, verbose = False):
        if not self.results:
            summary = {"mean_iou": 0.0, "count": 0}
        else:
            summary = {
                "mean_iou": sum(self.results) / len(self.results),
                "count": len(self.results), 
            }
        if verbose:
            summary["details"] = self.detailed_results if self.detailed_results else {}

        return summary
    
    def dump_config(self):
        return {
            "name": self.name,
            "ground_truth_path": str(self.ground_truth_path),
            "threshold": self.relevancy_threshold
        }