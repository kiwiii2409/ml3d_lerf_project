import json
import os
from pathlib import Path

import numpy as np
import torch
from lerf.data.utils.feature_dataloader import FeatureDataloader
from lerf.data.utils.patch_embedding_dataloader import PatchEmbeddingDataloader
from lerf.encoders.image_encoder import BaseImageEncoder
from tqdm import tqdm


class PyramidEmbeddingDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: BaseImageEncoder,
        image_list: torch.Tensor = None,
        cache_path: str = None,
        sam_loader = None
    ):
        assert "tile_size_range" in cfg
        assert "tile_size_res" in cfg
        assert "stride_scaler" in cfg
        assert "image_shape" in cfg
        assert "model_name" in cfg

        self.tile_sizes = torch.linspace(*cfg["tile_size_range"], cfg["tile_size_res"]).to(device)
        self.strider_scaler_list = [self._stride_scaler(tr.item(), cfg["stride_scaler"]) for tr in self.tile_sizes]

        self.model = model
        self.embed_size = self.model.embedding_dim
        self.data_dict = {}

        self.sam_loader = sam_loader # Mod1: added sam_loader as param
        self.mask_cache = {} # Mod1: cache to avoid repeated storage access for union masks

        super().__init__(cfg, device, image_list, cache_path)

    # Mod1: create mask value for each img_point and pass to scales function
    def __call__(self, img_points, scale=None):
        masks = self._get_ray_mask(img_points)

        if scale is None:
            return self._random_scales(img_points, masks)
        else:
            return self._uniform_scales(img_points, scale, masks)

    def _stride_scaler(self, tile_ratio, stride_scaler):
        return np.interp(tile_ratio, [0.05, 0.15], [1.0, stride_scaler])

    def load(self):
        # don't create anything, PatchEmbeddingDataloader will create itself
        cache_info_path = self.cache_path.with_suffix(".info")

        # check if cache exists
        if not cache_info_path.exists():
            raise FileNotFoundError

        # if config is different, remove all cached content
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            for f in os.listdir(self.cache_path):
                os.remove(os.path.join(self.cache_path, f))
            raise ValueError("Config mismatch")

        raise FileNotFoundError  # trigger create

    def create(self, image_list):
        os.makedirs(self.cache_path, exist_ok=True)
        for i, tr in enumerate(tqdm(self.tile_sizes, desc="Scales")):
            stride_scaler = self.strider_scaler_list[i]
            self.data_dict[i] = PatchEmbeddingDataloader(
                cfg={
                    "tile_ratio": tr.item(),
                    "stride_ratio": stride_scaler,
                    "image_shape": self.cfg["image_shape"],
                    "model_name": self.cfg["model_name"],
                },
                device=self.device,
                model=self.model,
                image_list=image_list,
                cache_path=Path(f"{self.cache_path}/level_{i}.npy"),
                sam_loader = self.sam_loader # Mod1: pass dataloader to CLIP 
            )

    def save(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))
        # don't save anything, PatchEmbeddingDataloader will save itself
        pass

    # Mod1: add mask values for each ray to data_dict calls, masks shape (B,)
    def _random_scales(self, img_points, masks):
        # img_points: (B, 3) # (img_ind, x, y)
        # return: (B, 512), some random scale (between 0, 1)
        img_points = img_points.to(self.device)
        random_scale_bin = torch.randint(self.tile_sizes.shape[0] - 1, size=(img_points.shape[0],), device=self.device)
        random_scale_weight = torch.rand(img_points.shape[0], dtype=torch.float16, device=self.device)

        stepsize = (self.tile_sizes[1] - self.tile_sizes[0]) / (self.tile_sizes[-1] - self.tile_sizes[0])

        bottom_interp = torch.zeros((img_points.shape[0], self.embed_size), dtype=torch.float16, device=self.device)
        top_interp = torch.zeros((img_points.shape[0], self.embed_size), dtype=torch.float16, device=self.device)

        for i in range(len(self.tile_sizes) - 1):
            ids = img_points[random_scale_bin == i]
            sub_masks = masks[random_scale_bin==i] # masks matching ids
            bottom_interp[random_scale_bin == i] = self.data_dict[i](ids, masks=sub_masks) # pass mask
            top_interp[random_scale_bin == i] = self.data_dict[i + 1](ids, masks=sub_masks)

        return (
            torch.lerp(bottom_interp, top_interp, random_scale_weight[..., None]),
            (random_scale_bin * stepsize + random_scale_weight * stepsize)[..., None],
        )

    # Mod1: add mask values for each ray to data_dict calls, masks shape (B,)
    def _uniform_scales(self, img_points, scale, masks):
        scale_bin = torch.floor(
            (scale - self.tile_sizes[0]) / (self.tile_sizes[-1] - self.tile_sizes[0]) * (self.tile_sizes.shape[0] - 1)
        ).to(torch.int64)
        scale_weight = (scale - self.tile_sizes[scale_bin]) / (
            self.tile_sizes[scale_bin + 1] - self.tile_sizes[scale_bin]
        )
        interp_lst = torch.stack([interp(img_points, masks=masks) for interp in self.data_dict.values()]) # pass mask to embed dataloader
        point_inds = torch.arange(img_points.shape[0])
        interp = torch.lerp(
            interp_lst[scale_bin, point_inds], 
            interp_lst[scale_bin + 1, point_inds],
            torch.Tensor([scale_weight]).half().to(self.device)[..., None],
        )
        return interp / interp.norm(dim=-1, keepdim=True), scale

    # Mod1: function to fetch a mask values (object=1,bgr=0) for a given ray
    def _get_ray_mask(self, img_points):
        """
        function to fetch a mask values (object=1,bgr=0) for a batch of given rays
        
        :param img_points: Tensor with shape (B, 3) storing B times (img_ind, x, y)
        :returns an array of (B,) where each index i indicates whether ray i hit the object=1 or the background=0

        """
        mask_accumulator = torch.zeros(img_points.shape[0], device=self.device) # gathers indicator for each ray

        sam_loader = self.sam_loader
        images = torch.unique(img_points[:, 0]) # get all unique image indices 
        for img_idx_tensor in images:
            img_idx = int(img_idx_tensor.item()) # convert tensor to int to avoid potential problems
            if img_idx in self.mask_cache:
                union_mask = self.mask_cache[img_idx]
            else:
                union_mask = sam_loader.get_masks_for_image(img_idx).float() # yields h x w union-over-masks
                self.mask_cache[img_idx] = union_mask
            rays_idx = (img_points[:, 0] == img_idx) # get all rays belonging to current image
            points = img_points[rays_idx] # n x 3 with N being the number of rays belonging to curr img

            points_x = points[:,1].long().clamp(0, union_mask.shape[0]-1).cpu()
            points_y = points[:,2].long().clamp(0, union_mask.shape[1]-1).cpu()

            img_masks = union_mask[points_x, points_y] # indexing [row,col]

            print(f"mask: {img_masks.shape}")
            print(f"rays idx: {rays_idx.shape}")
            print(f"mask_acc: {mask_accumulator.shape}")
            mask_accumulator[rays_idx] = img_masks.to(self.device)

        return mask_accumulator