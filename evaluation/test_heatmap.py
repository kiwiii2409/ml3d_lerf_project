import numpy as np 
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm 
import torch
from test_base import BaseTest

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except ImportError:
    print("Segement anything not installed")



class Heatmap_Test(BaseTest):
    def __init__(self, name:str = "Heatmap-SAM-IoU", ground_truth_path: Optional[Path] = None, relevancy_threshold: float = 0.5):
        self.relevancy_threshold = relevancy_threshold
        super().__init__(name, ground_truth_path)

        self.results = [] # store iou per image per prompt
        self.detailed_results = {}
        self.ground_truth = {} # Stores { '00177': List[Tensor_Masks] }

        self.sam_loaded = False
        self.mask_generator = None



    def _init_sam(self):
        if self.sam_loaded:
            return
        sam_ckpt = "/cluster/51/kzhou/sam_weights/sam_vit_h_4b8939.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
        sam.to(device=device)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            min_mask_region_area=100, # remove small disconnected patches (smaller than 100)
        )
        self.sam_loaded = True

    # TODO:(cahnge this to _get_ground_truth) to call sam on list of images and store in dictionary {filename: mask_dict}
    # keep sam masks separate (dont merge to object) and calcualte iou with all masks => find best fit as final iou value 
    # otherwise querying a single object in a scene with multiple objects will always lead to bad performance, as union_obj_mask is larger than single object


    # gets: {00177: image_data}
    def load_ground_truth(self, image_dict: Dict):
        self._init_sam()

        print(f"Generating SAM masks for {len(image_dict)} evaluation images...")
        for image_idx, image in tqdm(image_dict.items(), desc="sam_eval"):
            # TODO: run sam and store in self.ground_truth. again as dict with {image_idx}
            masks = self.mask_generator.generate(image)
            if len(masks) > 0:
                
                mask_stack = np.stack([m['segmentation'] for m in masks])# (num_masks x h x w)
                self.ground_truth[image_idx] = torch.from_numpy(mask_stack)
            else:
                self.ground_truth[image_idx] = None
        print("SAM generation complete")
        
    def evaluate_image(self, image_idx: str, rendered_layers: dict, model, gt_meta: dict):
        sam_masks = self.ground_truth.get(image_idx)        

        target_prompts = gt_meta["positives"]

        for prompt in target_prompts:
             # index of prompt in positives list == relevacy layer
            promtp_idx = model.image_encoder.positives.index(prompt)
            relevancy_key = f"relevancy_{promtp_idx}"
            relevancy_map = rendered_layers[relevancy_key].squeeze() 
            pred_mask = (relevancy_map > self.relevancy_threshold).float() # shape HxW

            print(f"Image Idx: {image_idx} with prompt: {prompt}")
            print(f"Shape PredMask: {pred_mask.shape}")
            print(f"Shape SamMask: {sam_masks[0].shape}")
            if sam_masks[0].shape != pred_mask.shape: # not sure if needed
                raise Exception(f"Invalid shape! sam_mask: {sam_masks[0].shape} vs. pred_mask: {pred_mask.shape}")
            
            sam_masks_gpu = (sam_masks).float().to(pred_mask.device)

            pred_mask = pred_mask.unsqueeze(0) # 1xHxW

            intersection = (pred_mask * sam_masks_gpu).sum(dim=(1,2)) # take sum per mask (num_mask,)
            union = torch.max(pred_mask, sam_masks_gpu).sum(dim=(1, 2)) # (num_mask,)

            ious = intersection / (union + 1e-8) # iou for each mask vs. predicted mask
            
            best_iou = torch.max(ious).item() # select max
            self.results.append(best_iou)

            if image_idx not in self.detailed_results:
                self.detailed_results[image_idx] = {}
            self.detailed_results[image_idx][prompt] = best_iou
        
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
            # "ground_truth_path": str(self.ground_truth_path), # no need as we generate our own gt here
            "threshold": self.relevancy_threshold
        }