global_flash_attn = False
global_layernorm = True
global_xformers = False
micro_frame_size = None

vae_out_channels = 16

model = dict(
    type="STDiT3-XL/2",
    qk_norm=True,
    pred_sigma=False,
    enable_flash_attn=True and global_flash_attn,
    enable_layernorm_kernel=True and global_layernorm,
    enable_xformers=False and global_xformers,
    uncond_cam_in_dim=(3, 7),
    cam_encoder_cls="src.models.stdit.control_embedder.CamEmbedder",
    cam_encoder_param=dict(
        input_dim=3,
        num=7,
        after_proj=True,
    ),
    bbox_embedder_cls="src.models.stdit.control_embedder.BBoxEmbedder",
    bbox_embedder_param=dict(
        classes=["car", "truck", "bus", "utility", "person", "child", "obstacle", "traffic sign"],
        class_token_dim=1152,
        embedder_num_freq=4,
        proj_dims=[1152, 512, 512, 1152],
        after_proj=True,
    ),
)

vae = dict(
    type="VideoAutoencoderKLCogVideoX",
    from_pretrained="./checkpoints/CogVideoX-2b",
    subfolder="vae",
    micro_frame_size=micro_frame_size,
    micro_batch_size=1,
    local_files_only=True
)

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
    num_sampling_steps=30,
        cfg_scale=1.0,
)
