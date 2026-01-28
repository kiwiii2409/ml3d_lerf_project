import json

import numpy as np
import torch
from lerf.data.utils.feature_dataloader import FeatureDataloader
from lerf.encoders.image_encoder import BaseImageEncoder
from tqdm import tqdm


class PatchEmbeddingDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: BaseImageEncoder,
        image_list: torch.Tensor = None,
        cache_path: str = None,
        sam_loader = None
    ):
        assert "tile_ratio" in cfg
        assert "stride_ratio" in cfg
        assert "image_shape" in cfg
        assert "model_name" in cfg

        self.kernel_size = int(cfg["image_shape"][0] * cfg["tile_ratio"])
        self.stride = int(self.kernel_size * cfg["stride_ratio"])
        self.padding = self.kernel_size // 2
        self.center_x = (
            (self.kernel_size - 1) / 2
            - self.padding
            + self.stride
            * np.arange(
                np.floor((cfg["image_shape"][0] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
            )
        )
        self.center_y = (
            (self.kernel_size - 1) / 2
            - self.padding
            + self.stride
            * np.arange(
                np.floor((cfg["image_shape"][1] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
            )
        )
        self.center_x = torch.from_numpy(self.center_x).half()
        self.center_y = torch.from_numpy(self.center_y).half()
        self.start_x = self.center_x[0].float()
        self.start_y = self.center_y[0].float()

        self.model = model
        self.embed_size = self.model.embedding_dim

        self.sam_loader = sam_loader # Mod1 added sam_loader as param

        super().__init__(cfg, device, image_list, cache_path)

    def load(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        if not cache_info_path.exists():
            raise FileNotFoundError
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            raise ValueError("Config mismatch")
        self.data = torch.from_numpy(np.load(self.cache_path)).half()

    def create(self, image_list):
        assert self.model is not None, "model must be provided to generate features"
        assert image_list is not None, "image_list must be provided to generate features"

        unfold_func = torch.nn.Unfold(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        ).to(self.device)

        img_embeds = []
        sam_loader = self.sam_loader
        for idx, img in enumerate(tqdm(image_list, desc="Embedding images", leave=False)):
            union_mask = sam_loader.get_masks_for_image(idx)
            union_mask = union_mask.to(self.device)
            
            img_embeds.append(self._embed_clip_tiles(img.unsqueeze(0), union_mask, unfold_func))
        self.data = torch.from_numpy(np.stack(img_embeds)).half()

    # Mod1: add masks as argument to call
    def __call__(self, img_points, masks):
        # img_points: (B, 3) # (img_ind, x, y) (img_ind, row, col)
        # return: (B, 512)
        img_points = img_points.cpu()
        masks = masks.cpu() # (B,) list of indices with 1=obj, 0=bgr

        img_ind, img_points_x, img_points_y = img_points[:, 0], img_points[:, 1], img_points[:, 2]

        x_ind = torch.floor((img_points_x - (self.start_x)) / self.stride).long()
        y_ind = torch.floor((img_points_y - (self.start_y)) / self.stride).long()
        return self._interp_inds(img_ind, x_ind, y_ind, img_points_x, img_points_y, masks)

    # Mod1: added mask_layer to indexing to retrieve correct embedding
    def _interp_inds(self, img_ind, x_ind, y_ind, img_points_x, img_points_y, masks):
        img_ind = img_ind.to(self.data.device)  # self.data is on cpu to save gpu memory, hence this line
        
        # if mask = 1 (object) -> layer 0 for obj_embeds, else take bgr_embeds
        mask_layer = (1-masks).long().to(self.data.device)

        topleft = self.data[img_ind, x_ind, y_ind, mask_layer].to(self.device)
        topright = self.data[img_ind, x_ind + 1, y_ind, mask_layer].to(self.device)
        botleft = self.data[img_ind, x_ind, y_ind + 1, mask_layer].to(self.device)
        botright = self.data[img_ind, x_ind + 1, y_ind + 1, mask_layer].to(self.device)

        x_stride = self.stride
        y_stride = self.stride
        right_w = ((img_points_x - (self.center_x[x_ind])) / x_stride).to(self.device)  # .half()
        top = torch.lerp(topleft, topright, right_w[:, None])
        bot = torch.lerp(botleft, botright, right_w[:, None])

        bot_w = ((img_points_y - (self.center_y[y_ind])) / y_stride).to(self.device)  # .half()
        return torch.lerp(top, bot, bot_w[:, None])

    # Mod1: change parameters, create separate embeddings for object and background
    def _embed_clip_tiles(self, image, union_mask, unfold_func):
        # create union of masks for object and background
        union_mask = union_mask.unsqueeze(0) # turns h x w into 1 x h x w (important for compariing with patches later, requires n x 1x h x w)
        union_obj_mask = union_mask.float() 
        union_bgr_mask = 1 - union_obj_mask


        # image augmentation: slow-ish (0.02s for 600x800 image per augmentation)
        aug_imgs = torch.cat([image])

        # unfolded image patches (1x 3 x h x w) -> (n_patches x 3 x k x k)
        tiles = unfold_func(aug_imgs).permute(2, 0, 1).reshape(-1, 3, self.kernel_size, self.kernel_size).to("cuda")

        # unfolded object and background mask patches (1 x h x w) -> (n_patches x 1 x k x k)
        obj_mask_tiles = unfold_func(union_obj_mask.unsqueeze(0)).permute(2, 0, 1).reshape(-1, 1, self.kernel_size, self.kernel_size).to("cuda")
        bgr_mask_tiles = unfold_func(union_bgr_mask.unsqueeze(0)).permute(2, 0, 1).reshape(-1, 1, self.kernel_size, self.kernel_size).to("cuda")
        
        # apply masks, output: (n_patches x 3 x k x k)
        obj_tiles = tiles * obj_mask_tiles
        bgr_tiles = tiles* bgr_mask_tiles

        # merge into batch  (2*n_patches x 3 x k x k)
        combined_tiles = torch.cat([obj_tiles, bgr_tiles], dim=0).to(self.device)
        with torch.no_grad():
            combined_embeds = self.model.encode_image(combined_tiles) # (2* n_patches x 512)
        combined_embeds /= combined_embeds.norm(dim=-1, keepdim=True)

        num_tiles = tiles.shape[0]
        obj_embeds = combined_embeds[:num_tiles] #(n_patches x 512)
        bgr_embeds = combined_embeds[num_tiles:] #(n_patches x 512)

        # reshape list of embeddings to grid of embeddings for each tile (h_grid + 1 x w_grid  + 1 x 512)
        def reshape_to_grid(embeds): 
            embeds = embeds.reshape((self.center_x.shape[0], self.center_y.shape[0], -1)) #(h_grid x w_grid x512)
            embeds = torch.concat((embeds, embeds[:, [-1], :]), dim=1) # add padding to make interpolation possible
            embeds = torch.concat((embeds, embeds[[-1], :, :]), dim=0)
            return embeds
        
        obj_grid = reshape_to_grid(obj_embeds)
        bgr_grid = reshape_to_grid(bgr_embeds)
        return torch.stack([obj_grid, bgr_grid], dim=2).detach().cpu().numpy() # two layers, with obj at layer 0, bgr at layer 1
