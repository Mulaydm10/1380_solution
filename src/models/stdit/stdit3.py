import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from transformers import PretrainedConfig, PreTrainedModel


from src.models.layers.blocks import (
    Attention,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_layernorm,
    t2i_modulate,
)
from src.registry import MODELS

from .utils import load_module


class STDiT3Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flash_attn=False,
        enable_xformers=False,
        enable_layernorm_kernel=False,
        rope=None,
        qk_norm=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn

        self.norm1 = get_layernorm(
            hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel
        )

        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
            enable_flash_attn=enable_flash_attn,
            enable_xformers=enable_xformers,
        )

        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)
        self.bev_cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)

        self.norm2 = get_layernorm(
            hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel
        )
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approx_gelu,
            drop=0,
        )

        # other helpers
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(
        self,
        x,
        y,
        t,  # this t
        T=None,  # number of frames
        S=None,  # number of pixel patches
        NC=None,  # number of cameras
    ):

        B, N, C = x.shape
        assert (N == T * S) and (B % NC == 0)
        b = B // NC

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = repeat(
            self.scale_shift_table[None] + t.reshape(b, 6, -1),
            "b ... -> (b NC) ...",
            NC=NC,
        ).chunk(6, dim=1)

        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        ######################
        # attention
        ######################

        x_m = rearrange(x_m, "(B NC) (T S) C -> (B T) (NC S) C", NC=NC, T=T, S=S)
        x_m = self.attn(x_m)
        x_m = rearrange(x_m, "(B T) (NC S) C -> (B NC) (T S) C", NC=NC, T=T, S=S)

        # modulate (attention)
        x_m_s = gate_msa * x_m

        # residual
        x = x + self.drop_path(x_m_s)

        ######################
        # cross attn
        ######################
        x_c = self.cross_attn(x, y[:, 0], None)

        x = x + x_c

        # BEV cross attn
        if y.shape[1] > 1:
            x_bev = self.bev_cross_attn(x, y[:, 1], None)
            x = x + x_bev

        ######################
        # MLP
        ######################
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m

        # residual
        x = x + self.drop_path(x_m_s)
        return x


class STDiT3Config(PretrainedConfig):
    model_type = "STDiT3"

    def __init__(
        self,
        input_size=(1, 32, 32),
        input_sq_size=512,
        in_channels=16,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path: float = 0.0,
        qk_norm=True,
        enable_flash_attn=False,
        enable_xformers=False,
        enable_layernorm_kernel=False,
        qk_norm_trainable=False,
        uncond_cam_in_dim=(3, 7),
        cam_encoder_cls=None,
        cam_encoder_param={},
        bbox_embedder_cls=None,
        bbox_embedder_param={},
        **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.qk_norm_trainable = qk_norm_trainable
        self.enable_xformers = enable_xformers
        self.uncond_cam_in_dim = uncond_cam_in_dim
        self.cam_encoder_cls = cam_encoder_cls
        self.cam_encoder_param = cam_encoder_param
        self.bbox_embedder_cls = bbox_embedder_cls
        self.bbox_embedder_param = bbox_embedder_param
        super().__init__(**kwargs)


class STDiT3(PreTrainedModel):
    """
    Diffusion model with a Transformer backbone.
    """

    config_class = STDiT3Config

    def __init__(self, config: STDiT3Config):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels

        # model size related
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # computation related
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_xformers = config.enable_xformers
        self.enable_layernorm_kernel = config.enable_layernorm_kernel

        # input size related
        self.patch_size = config.patch_size
        self.input_sq_size = config.input_sq_size
        self.pos_embed = PositionEmbedding2D(self.hidden_size)
        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)

        # embedding
        self.x_embedder = PatchEmbed3D(self.patch_size, self.in_channels, self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True)
        )

        # base_token, should not be trainable
        self.register_buffer("base_token", torch.randn(self.hidden_size))
        # init unified control embedder
        from .control_embedder import ControlEmbedder
        self.control_embedder = ControlEmbedder(config)

        # base blocks
        drop_path = [x.item() for x in torch.linspace(0, config.drop_path, self.depth)]
        self.spatial_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flash_attn=self.enable_flash_attn,
                    enable_xformers=self.enable_xformers,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    qk_norm=config.qk_norm,
                )
                for i in range(self.depth)
            ]
        )

        # final layer
        self.final_layer = T2IFinalLayer(
            self.hidden_size, np.prod(self.patch_size), self.out_channels
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        def _zero_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.constant_(module.weight, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # zero init embedder proj
        if hasattr(self, 'control_embedder') and self.control_embedder is not None:
            if self.control_embedder.bbox_embedder is not None and hasattr(self.control_embedder.bbox_embedder, 'after_proj'):
                _zero_init(self.control_embedder.bbox_embedder.after_proj)
            if self.control_embedder.cam_embedder is not None and hasattr(self.control_embedder.cam_embedder, 'after_proj'):
                _zero_init(self.control_embedder.cam_embedder.after_proj)
            if self.control_embedder.cam_embedder is not None and hasattr(self.control_embedder.cam_embedder, 'emb2token'):
                nn.init.normal_(self.control_embedder.cam_embedder.emb2token.weight, std=0.02)
        else:
            # Fallback for original structure if control_embedder not present
            if self.bbox_embedder is not None:
                _zero_init(self.bbox_embedder.after_proj)
            if self.camera_embedder is not None:
                _zero_init(self.camera_embedder.after_proj)

            if self.camera_embedder is not None:
                nn.init.normal_(self.camera_embedder.emb2token.weight, std=0.02)

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def encode_box(self, bboxes, drop_mask, bbox_mask=None):  # changed
        B, T, seq_len = bboxes["bboxes"].shape[:3]
        bbox_embedder_kwargs = {}
        for k, v in bboxes.items():
            bbox_embedder_kwargs[k] = v.clone()

        drop_mask = repeat(drop_mask, "B T -> B T S", S=seq_len)
        _null_mask = torch.ones_like(bbox_embedder_kwargs["masks"])
        _null_mask[bbox_embedder_kwargs["masks"] == 0] = 0
        _mask = torch.ones_like(bbox_embedder_kwargs["masks"])
        _mask[bbox_embedder_kwargs["masks"] == -1] = 0
        _mask[
            torch.logical_and(
                bbox_embedder_kwargs["masks"] == 1,
                drop_mask == 0,  # only drop those real boxes
            )
        ] = 0

        bbox_emb = self.bbox_embedder(
            bboxes=bbox_embedder_kwargs["bboxes"],
            classes=bbox_embedder_kwargs["classes"].type(torch.int32),
            null_mask=_null_mask,
            mask=bbox_mask,  # Use the passed bbox_mask
            box_latent=bbox_embedder_kwargs.get("box_latent", None),
        )
        return bbox_emb

    def encode_cam(self, cam, embedder, drop_mask):
        B, T, S = cam.shape[:3]
        NC = B // drop_mask.shape[0]
        mask = repeat(drop_mask, "b T -> (b NC T S)", NC=NC, S=S)
        cam = rearrange(cam, "B T S ... -> (B T S) ...")
        cam_emb, _ = embedder.embed_cam(cam, mask, T=T, S=S)
        return cam_emb

    def encode_cond_sequence(self, bbox, cams, bev_grid, drop_cond_mask, NC):
        # All embedding logic is now in ControlEmbedder
        cond_tokens = self.control_embedder(
            bboxes_dict=bbox,
            camera_params=cams,
            bev_grid=bev_grid,
            # The mask for bboxes will be extracted from the bbox dict inside the embedder
        )
        return cond_tokens, None

    def forward(self, x, t, encoder_hidden_states=None, **kwargs):
        drop_cond_mask = None
        """
        Forward pass
        """
        dtype = self.x_embedder.proj.weight.dtype
        B = x.size(0)
        if drop_cond_mask is None:  # camera
            drop_cond_mask = torch.ones((B), device=x.device, dtype=x.dtype)

        x = x.to(dtype)

        cond_cam = rearrange(cond_cam, "(B NC) C T ... -> B NC C T ...", NC=1)
        x = torch.cat([x[:, :3], cond_cam, x[:, 3:]], dim=1)
        x = rearrange(x, "B NC C T ... -> (B NC) C T ...", NC=NC)
        timestep = timestep.to(dtype)

        # === get pos embed ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        S = H * W

        base_size = round(S**0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        t_mlp = self.t_block(t)

        # === get y embed ===
        y, y_lens = self.encode_cond_sequence(bbox, cams, drop_cond_mask, NC)  # (B, L, D)

        # === get x embed ===
        x_b = self.x_embedder(x)  # [B, N, C]
        x_b = rearrange(x_b, "B (T S) C -> B T S C", T=T, S=S)
        x = x_b + pos_emb

        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === blocks ===

        for block_i in range(0, self.depth):
            x = self.spatial_blocks[block_i](
                x,
                y,
                t_mlp,
                T,
                S,
                NC,
            )

        # === final layer ===
        x = self.final_layer(
            x,
            repeat(t, "b d -> (b NC) d", NC=NC),
            T=T,
            S=S,
        )
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        x = x.to(torch.float32)
        x = rearrange(x, "(B NC) C T ... -> B NC C T ...", NC=NC)
        x = torch.cat([x[:, :3], x[:, 4:]], dim=1)
        x = rearrange(x, "B NC C T ... -> B (C NC) T ...", NC=(NC - 1))
        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x


@MODELS.register_module("STDiT3-XL/2")
def SdgSTDiT3_XL_2(**kwargs):
    config = STDiT3Config(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
    model = STDiT3(config)
    return model
