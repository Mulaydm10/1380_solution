import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .utils import zero_module
import timm

# --- Original Embedders (Restored) ---

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

class BBoxEmbedder(nn.Module):
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
        output_num = 8
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
        self.null_pos_feature = torch.nn.Parameter(torch.zeros([self.fourier_embedder.out_dim * output_num]))
        self.mask_class_feature = torch.nn.Parameter(torch.zeros([class_token_dim]))
        self.mask_pos_feature = torch.nn.Parameter(torch.zeros([self.fourier_embedder.out_dim * output_num]))
        if after_proj:
            self.after_proj = zero_module(nn.Linear(proj_dims[-1], proj_dims[-1]))
        else:
            self.after_proj = None

    def forward_feature(self, pos_emb, cls_emb):
        emb = self.bbox_proj(pos_emb)
        emb = F.silu(emb)
        emb = torch.cat([emb, cls_emb], dim=-1)
        emb = self.second_linear(emb)
        return emb

    def forward(self, bboxes, classes, null_mask=None, mask=None, **kwargs):
        # Robust squeeze for pad dims (drop all 1s; handle 4D+)
        if bboxes.dim() > 3:
            bboxes = bboxes.squeeze()
        if classes.dim() > 3:
            classes = classes.squeeze()
        if null_mask is not None and null_mask.dim() > 3:
            null_mask = null_mask.squeeze()
        if mask is not None and mask.dim() > 3:
            mask = mask.squeeze()

        # Ensure 3D for unpack (B, T, N) where N=1 for class ID/mask
        if len(classes.shape) == 2:
            classes = classes.unsqueeze(-1)  # [B, max, 1]
        if null_mask is not None and len(null_mask.shape) == 2:
            null_mask = null_mask.unsqueeze(-1)  # [B, max, 1]
        if mask is not None and len(mask.shape) == 2:
            mask = mask.unsqueeze(-1)  # [B, max, 1]

        B, T, N = classes.shape
        bboxes = rearrange(bboxes, "b n ... -> (b n) ...")

        def handle_none_mask(_mask):
            if _mask is None:
                _mask = torch.ones(len(bboxes), device=bboxes.device)
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

        cls_emb = self._class_tokens[classes.flatten()]
        cls_emb = cls_emb * null_mask + self.null_class_feature[None] * (1 - null_mask)
        cls_emb = cls_emb * mask + self.mask_class_feature[None] * (1 - mask)

        emb = self.forward_feature(pos_emb, cls_emb)
        emb = rearrange(emb, "(b n) ... -> b n ...", n=N)
        if self.after_proj:
            emb = self.after_proj(emb)
        emb = rearrange(emb, "(b t) n d -> b t n d", t=T)
        return emb

class CamEmbedder(nn.Module):
    def __init__(self, input_dim, out_dim, num=7, num_freqs=4, include_input=True, log_sampling=True, after_proj=False):
        super().__init__()
        self.embedder = get_embedder(input_dim, num_freqs, include_input, log_sampling)
        self.emb2token = nn.Linear(self.embedder.out_dim * num, out_dim)
        self.uncond_cam = torch.nn.Parameter(torch.randn([input_dim, num]))
        if after_proj:
            self.after_proj = zero_module(nn.Linear(out_dim, out_dim))
        else:
            self.after_proj = None

    def embed_cam(self, param, mask=None, **kwargs):
        if param.shape[1] == 4:
            param = param[:, :-1]
        (bs, C_param, emb_num) = param.shape
        assert C_param == 3
        if mask is not None:
            param = torch.where((mask > 0)[:, None, None], param, self.uncond_cam[None])
        emb = self.embedder(rearrange(param, "b d c -> (b c) d"))
        emb = rearrange(emb, "(b c) d -> b (c d)", b=bs)
        token = self.emb2token(emb)
        if self.after_proj:
            token = self.after_proj(token)
        return token, emb

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Please call other functions.")

# --- New BEV Embedder ---

class BEVEmbedder(nn.Module):
    def __init__(self, embed_dim=1152):
        super().__init__()
        self.cnn = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0, features_only=True)
        self.proj = nn.Linear(1280, embed_dim)

    def forward(self, bev_grid):
        features = self.cnn(bev_grid[:, :3, :, :])[-1]
        features = features.flatten(2).transpose(1, 2)
        tokens = self.proj(features)
        return tokens

# --- New Unified Control Embedder ---

class ControlEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bbox_embedder = BBoxEmbedder(**config.bbox_embedder_param)
        cam_params = getattr(config, 'cam_embedder_param', {'input_dim': 3, 'out_dim': config.hidden_size, 'num': 7, 'after_proj': True})
        self.cam_embedder = CamEmbedder(**cam_params)
        self.bev_embedder = BEVEmbedder(embed_dim=config.hidden_size)

    def forward(self, bboxes_dict, camera_params, bev_grid=None, **kwargs):
        # Full squeeze for all leading 1s (robust to pad dims)
        bbox_data = bboxes_dict['bboxes']['data'].squeeze()  # Drops all dim=1
        class_data = bboxes_dict['classes']['data'].squeeze()  # [B, max]
        attention_mask = bboxes_dict['bboxes']['mask'].squeeze(1).squeeze(1).unsqueeze(1).float()  # [B, 1, max_objs]

        # Reshape to 3D for original unpack "b t n"
        if bbox_data.dim() == 3: # If batch size is 1, squeeze might remove it. Add it back.
            bbox_data = bbox_data.unsqueeze(0)
            class_data = class_data.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            
        class_data = class_data.unsqueeze(-1)  # [B, max, 1]
        attention_mask = attention_mask.unsqueeze(-1)  # [B, max, 1]

        # Call with 3D masks
        bbox_tokens = self.bbox_embedder(
            bboxes=bbox_data,
            classes=class_data,
            mask=attention_mask,
            null_mask=(1 - attention_mask.squeeze(-1)).unsqueeze(-1),
            **kwargs
        )
        
        # Cam stream (original)
        cam_tokens, _ = self.cam_embedder.embed_cam(camera_params)
        
        # BEV stream (new; optional)
        if bev_grid is not None:
            bev_tokens = self.bev_embedder(bev_grid)
            all_tokens = torch.cat([bbox_tokens, cam_tokens.unsqueeze(1), bev_tokens], dim=1)
        else:
            all_tokens = torch.cat([bbox_tokens, cam_tokens.unsqueeze(1)], dim=1)
        
        return all_tokens