import numpy as np 
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm 
import torch
import cv2
from test_base import BaseTest

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except ImportError:
    print("Segement anything not installed")



class Heatmap_Test(BaseTest):
    def __init__(self, name:str = "Heatmap-SAM-IoU", ground_truth_path: Optional[Path] = None, output_path: Optional[Path] = None, relevancy_threshold: float = 0.5):
        super().__init__(name, ground_truth_path, output_path)

        self.relevancy_threshold = relevancy_threshold

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
        )
        self.sam_loaded = True


    # args: {00177: image_data}
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

            if sam_masks[0].shape != pred_mask.shape: # not sure if needed
                raise Exception(f"Invalid shape! sam_mask: {sam_masks[0].shape} vs. pred_mask: {pred_mask.shape}")
            
            sam_masks_gpu = (sam_masks).float().to(pred_mask.device)

            pred_mask = pred_mask.unsqueeze(0) # 1xHxW

            intersection = (pred_mask * sam_masks_gpu).sum(dim=(1,2)) # take sum per mask: (num_mask,)
            union = torch.max(pred_mask, sam_masks_gpu).sum(dim=(1, 2)) # (num_mask,)
            ious = intersection / (union + 1e-8) # iou for each mask vs. predicted mask
            
            # union over obj-masks leads to always bad results, as a multi-object scene which is queried for a single object will always have a low IOU
            # select max iou and idx (best fitting mask)
            max_tuple = torch.max(ious, dim=0)
            best_iou = max_tuple[0].item()
            best_idx = max_tuple[1].item()
            self.results.append(best_iou)

            if image_idx not in self.detailed_results:
                self.detailed_results[image_idx] = {}
            self.detailed_results[image_idx][prompt] = best_iou

            if self.output_path: # only visualize if output path is specified
                best_sam_mask = sam_masks[best_idx].cpu().numpy() # best mask as hxw boolean
                rgb_tensor = rendered_layers['rgb'].cpu().numpy() # get rgb image for overlay (range 0-1)
                bgr_image = cv2.cvtColor(rgb_tensor, cv2.COLOR_RGB2BGR) # convert for opencv processing
                self.visualize(image_idx, prompt, bgr_image, relevancy_map, best_sam_mask)

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
    
    # TODO: store sam_mask as image & heatmap overlayed on rgb image in folder of result. naming should be iamge_idx mask_idx of both files. adapt eval_lerf to store all results as 
    # - results/
    # -- 2026-01-28_14-26-45_lerf_mod1_bouquet_2026-01-17_192939/
    # --- summary.json
    # --- image_00001.jpg
    # --- mask_00001.jpg
    # --- image_00008.jpg
    # --- mask_00008.jpg
    # ...
    def visualize(self, image_idx:str, prompt: str, bgr_image, relevancy, sam_mask):  
        """
        Docstring for visualize
        
        :param image_idx: string of image
        :param prompt: annotation string for the image
        :param bgr_image: float numpy in bgr color space
        :param relevancy: float tensor, predicted relevancy map, hxw
        :param sam_mask: best fitting mask generated by sam
        """     
        # normalize to 0-255 int
        bgr_image_norm = (bgr_image * 255).astype(np.uint8) 
        relevancy_norm = (relevancy * 255).cpu().numpy().astype(np.uint8) 
        relevancy_color = cv2.applyColorMap(relevancy_norm, cv2.COLORMAP_JET)

        relevancy_mask = (relevancy > self.relevancy_threshold).unsqueeze(2).cpu().numpy() # zero out any non-relevant area
        relevancy_filtered = relevancy_color * relevancy_mask

        overlay_lerf = cv2.addWeighted(bgr_image_norm, 1, relevancy_filtered, 0.8, 0) # heatmap + img
        
        
        green_layer = np.zeros_like(bgr_image_norm)
        green_layer[sam_mask] = [0, 255, 0] # green inside of mask
        overlay_sam = cv2.addWeighted(bgr_image_norm, 1, green_layer, 0.4, 0) # sam_mask + img 


        heatmap_path = self.output_path / f"{image_idx}_{prompt}_image.jpg"
        sam_path = self.output_path / f"{image_idx}_{prompt}_mask.jpg"
        
        cv2.imwrite(str(heatmap_path), overlay_lerf)
        cv2.imwrite(str(sam_path), overlay_sam)