import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils import zero_module


import timm

class BEVEmbedder(nn.Module):
    def __init__(self, output_dim=1152):
        super().__init__()
        self.cnn = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0, features_only=True)
        # Example: EfficientNet-B0 produces [1280, 4, 4] features for a 256x256 input.
        # We need to project this to the model's hidden size.
        self.proj = nn.Linear(1280, output_dim)

    def forward(self, bev):
        # Input bev shape: [B, 5, 256, 256]
        # EfficientNet expects 3 channels, we will use the first 3 (occ, lanes, crosswalks)
        features = self.cnn(bev[:, :3, :, :])[-1] # Get features from last stage
        features = features.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, H*W, C]
        tokens = self.proj(features) # [B, H*W, output_dim]
        return tokens


class ControlEmbedder(nn.Module): # Assuming a ControlEmbedder class exists or should be created
    def __init__(self, bbox_embedder, cam_embedder, bev_embedder):
        super().__init__()
        self.bbox_embedder = bbox_embedder
        self.cam_embedder = cam_embedder
        self.bev_embedder = bev_embedder

    def forward(self, bboxes, cams, bev_grid=None, mask=None):
        bbox_tokens = self.bbox_embedder(bboxes['bboxes']['data'], bboxes['classes']['data'], mask=mask)
        cam_tokens = self.cam_embedder(cams)
        
        if bev_grid is not None and self.bev_embedder is not None:
            bev_tokens = self.bev_embedder(bev_grid)
            return torch.cat([bbox_tokens, cam_tokens, bev_tokens], dim=1)
        else:
            return torch.cat([bbox_tokens, cam_tokens], dim=1)
    def __init__(
        self,
        classes=["car", "truck", "bus", "utility", "person", "child", "obstacle", "traffic sign"],
        class_token_dim=768,
        embedder_num_freq=4,
        proj_dims=[768, 512, 512, 768],
        after_proj=False,
        **kwargs,
    ):
        super().__init__()
        self.classes = classes
        n_classes = len(classes)
        input_dims = 3
        output_num = 8  # 8 points

        self.fourier_embedder = get_embedder(input_dims, embedder_num_freq)
        self.bbox_proj = nn.Linear(self.fourier_embedder.out_dim * output_num, proj_dims[0])
        self.second_linear = nn.Sequential(
            nn.Linear(proj_dims[0] + class_token_dim, proj_dims[1]),
            nn.SiLU(),
            nn.Linear(proj_dims[1], proj_dims[2]),
            nn.SiLU(),
            nn.Linear(proj_dims[2], proj_dims[3]),
        )

        class_tokens = torch.randn(n_classes, class_token_dim)
        self.register_parameter("_class_tokens", nn.Parameter(class_tokens))
        self.null_class_feature = torch.nn.Parameter(torch.zeros([class_token_dim]))
        self.null_pos_feature = torch.nn.Parameter(
            torch.zeros([self.fourier_embedder.out_dim * output_num])
        )

        self.mask_class_feature = torch.nn.Parameter(torch.zeros([class_token_dim]))
        self.mask_pos_feature = torch.nn.Parameter(
            torch.zeros([self.fourier_embedder.out_dim * output_num])
        )

        if after_proj:
            self.after_proj = zero_module(nn.Linear(proj_dims[-1], proj_dims[-1]))
        else:
            self.after_proj = None

    def forward_feature(self, pos_emb, cls_emb):
        emb = self.bbox_proj(pos_emb)
        emb = F.silu(emb)

        # combine
        emb = torch.cat([emb, cls_emb], dim=-1)
        emb = self.second_linear(emb)
        return emb

    def forward(
        self,
        bboxes: torch.Tensor,
        classes: torch.LongTensor,
        null_mask=None,
        mask=None,
        box_latent=None,
        **kwargs,
    ):
        """Please do filter before input is needed.

        Args:
            bboxes (torch.Tensor): Expect (B, N, 8, 3)
            classes (torch.LongTensor): (B, N)
            null_mask: 0 -> null, 1 -> keep, really no box/padding
            mask: 0 -> mask, 1 -> keep, drop in any case

        Return:
            size B x N x emb_dim=768
        """
        B, T, N = classes.shape
        bboxes = rearrange(bboxes, "b t n ... -> (b t) n ...")
        classes = rearrange(classes, "b t n -> (b t) n")
        if box_latent is not None:
            box_latent = rearrange(box_latent, "b t n ... -> (b t) n ...")
        if null_mask is not None:
            null_mask = rearrange(null_mask, "b t n -> (b t) n")
        if mask is not None:
            mask = rearrange(mask, "b t n -> (b t) n")

        # (B, N) = classes.shape
        bboxes = rearrange(bboxes, "b n ... -> (b n) ...")

        def handle_none_mask(_mask):
            if _mask is None:
                _mask = torch.ones(len(bboxes))
            else:
                _mask = _mask.flatten()
            _mask = _mask.unsqueeze(-1).type_as(self.null_pos_feature)
            return _mask

        mask = handle_none_mask(mask)
        null_mask = handle_none_mask(null_mask)

        # box
        pos_emb = self.fourier_embedder(bboxes)

        pos_emb = pos_emb.reshape(pos_emb.shape[0], -1).type_as(self.null_pos_feature)
        pos_emb = pos_emb * null_mask + self.null_pos_feature[None] * (1 - null_mask)
        pos_emb = pos_emb * mask + self.mask_pos_feature[None] * (1 - mask)

        # class
        cls_emb = self._class_tokens[classes.flatten()]
        cls_emb = cls_emb * null_mask + self.null_class_feature[None] * (1 - null_mask)
        cls_emb = cls_emb * mask + self.mask_class_feature[None] * (1 - mask)

        # combine
        emb = self.forward_feature(pos_emb, cls_emb)
        emb = rearrange(emb, "(b n) ... -> b n ...", n=N)
        if self.after_proj:
            emb = self.after_proj(emb)

        emb = rearrange(emb, "(b t) n d -> b t n d", t=T)
        return emb


class FourierEmbedder:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(input_dims, num_freqs, include_input=True, log_sampling=True):
    embed_kwargs = {
        "input_dims": input_dims,
        "num_freqs": num_freqs,
        "max_freq_log2": num_freqs - 1,
        "include_input": include_input,
        "log_sampling": log_sampling,
        "periodic_fns": [torch.sin, torch.cos],
    }
    embedder_obj = FourierEmbedder(**embed_kwargs)
    logging.debug(f"embedder out dim = {embedder_obj.out_dim}")
    return embedder_obj


class CamEmbedder(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        num=7,
        num_freqs=4,
        include_input=True,
        log_sampling=True,
        after_proj=False,
    ):
        super().__init__()
        self.embedder = get_embedder(input_dim, num_freqs, include_input, log_sampling)
        self.emb2token = nn.Linear(self.embedder.out_dim * num, out_dim)
        logging.info(
            f"[{self.__class__.__name__}] init camera embedder with input_dim={input_dim}, num={num}."
        )
        self.uncond_cam = torch.nn.Parameter(torch.randn([input_dim, num]))
        if after_proj:
            self.after_proj = zero_module(nn.Linear(out_dim, out_dim))
        else:
            self.after_proj = None

    def embed_cam(self, param, mask=None, **kwargs):
        """
        Args:
            camera (torch.Tensor): [N, 3, num] or [N, 4, num]
        """
        if param.shape[1] == 4:
            param = param[:, :-1]
        (bs, C_param, emb_num) = param.shape
        assert C_param == 3

        # apply mask
        if mask is not None:
            param = torch.where((mask > 0)[:, None, None], param, self.uncond_cam[None])
        # embeding and project to token
        emb = self.embedder(rearrange(param, "b d c -> (b c) d"))
        emb = rearrange(emb, "(b c) d -> b (c d)", b=bs)
        token = self.emb2token(emb)
        if self.after_proj:
            token = self.after_proj(token)
        return token, emb

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Please call other functions.")
