import torch
from einops import repeat, rearrange


def move_to(obj, device, dtype=None, filter=lambda x: True):
    if torch.is_tensor(obj):
        if filter(obj):
            if dtype is None:
                dtype = obj.dtype
            return obj.to(device, dtype)
        else:
            return obj
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device, dtype, filter)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device, dtype, filter))
        return res
    elif obj is None:
        return obj
    else:
        raise TypeError(f"Invalid type {obj.__class__} for move_to.")


def prepare_control_data(
    batch: dict,
    vae,
    device,
    dtype,
    B: int,
    T: int,
    NC: int,
):
    model_args = {}
    bbox = batch.pop("bboxes_3d_data", None)  # B, T, NC=1, len, 8, 3

    if bbox is not None:
        for k, v in bbox.items():
            if k != "masks":
                v = repeat(v, "B T S ... -> B T (S NC) ...", NC=NC)
            bbox[k] = rearrange(v, "B T NC ... -> (B NC) T ...")  # BxNC, T, len, 3, 8

    cams = batch.pop("camera_param", None)  # B, T, NC, 3, 7
    if cams is not None:
        cams = rearrange(cams, "B T NC ... -> (B NC) T 1 ...")  # BxNC, T, 1, 3, 7

    cond_cam = batch.pop("cond_cam", None)
    if cond_cam is not None:
        cond_cam = rearrange(cond_cam, "B T NC C ... -> (B NC) C T ...").to(device, dtype)
        cond_cam = vae.encode(cond_cam)


    # == video meta info ==
    model_args["cond_cam"] = cond_cam
    model_args["bbox"] = bbox
    model_args["cams"] = cams
    model_args["height"] = batch.pop("height")
    model_args["width"] = batch.pop("width")
    model_args = move_to(model_args, device=device, dtype=dtype)
    # no need to move these
    model_args["B"] = B
    model_args["T"] = T
    model_args["NC"] = NC
    return model_args


def inference(
    model,
    scheduler,
    vae,
    model_args,
    device,
    dtype,
    generator=None,
):
    B, T, NC = model_args["B"], model_args["T"], model_args["NC"]
    latent_size = vae.get_latent_size(
        (T, int(model_args["height"][0]), int(model_args["width"][0]))
    )
    z = torch.randn(
        B,
        vae.out_channels * (model_args["NC"] - 1),
        *latent_size,
        generator=generator,
    ).to(device=device, dtype=dtype)

    cond_cam = model_args["cond_cam"].clone()
    # == add null condition ==
    if hasattr(model, "module"):
        model_ = model.module
    else:
        model_ = model

    if model_.camera_embedder is not None:
        uncond_cam = model_.camera_embedder.uncond_cam.to(device)
    else:
        uncond_cam = None

    _model_args = add_null_condition(
        model_args,
        uncond_cam,
        prepend=False,
    )

    # == inference ==
    torch.cuda.empty_cache()
    masks = None
    samples = scheduler.sample(
        model,
        z=z,
        device=device,
        model_args=_model_args,
        progress=True,
        mask=masks
    )

    samples = rearrange(samples, "B (C NC) T ... -> B NC C T ...", NC=(NC-1))
    samples = torch.cat([samples[:, :3], cond_cam[:, None], samples[:, 3:]], dim=1)
    samples = rearrange(samples, "B NC C T ... -> (B NC) C T ...", NC=NC)
    samples = vae.decode(samples.to(dtype), num_frames=T)
    samples = rearrange(samples, "(B NC) C T ... -> B NC C T ...", NC=NC)
    return samples


def add_null_condition(
    _model_args, uncond_cam, prepend=False):
    # will not change the original dict
    unchanged_keys = [
        "height",
        "width",
        "B",
        "T",
        "NC",
    ]
    handled_keys = []
    model_args = {}
    if "bbox" in _model_args and _model_args["bbox"] is not None:
        handled_keys.append("bbox")
        _bbox = _model_args["bbox"]
        bbox = {}
        for k in _bbox.keys():
            null_item = torch.zeros_like(_bbox[k])
            if prepend:
                bbox[k] = torch.cat([null_item, _bbox[k]], dim=0)
            else:
                bbox[k] = torch.cat([_bbox[k], null_item], dim=0)
        model_args["bbox"] = bbox
    else:
        handled_keys.append("bbox")
        model_args["bbox"] = None

    if "cams" in _model_args and _model_args["cams"] is not None:
        handled_keys.append("cams")
        cams = _model_args["cams"]  # BxNC, T, 1, 3, 7
        null_cams = torch.zeros_like(cams)
        BNC, T, L = null_cams.shape[:3]
        null_cams = null_cams.reshape(-1, 3, 7)
        null_cams[:] = uncond_cam[None]
        null_cams = null_cams.reshape(BNC, T, L, 3, 7)
        if prepend:
            model_args["cams"] = torch.cat([null_cams, cams], dim=0)
        else:
            model_args["cams"] = torch.cat([cams, null_cams], dim=0)
    else:
        handled_keys.append("cams")
        model_args["cams"] = None

    for k in _model_args.keys():
        if k in handled_keys:
            continue
        elif k in unchanged_keys:
            model_args[k] = _model_args[k]
        else:
            model_args[k] = repeat(_model_args[k], "b ... -> (2 b) ...")
    return model_args
