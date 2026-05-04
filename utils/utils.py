import random

import numpy as np
import torch
from PIL import Image

try:
    from multispectral_config import normalization_config
except Exception:
    # 兼容没有多光谱配置文件的旧流程：普通 RGB 图像仍然只做 /255。
    normalization_config = {
        "reflectance_scale": 10000.0,
        "enable_clip": False,
        "clip_min": None,
        "clip_max": None,
        "enable_mean_std": False,
        "mean": None,
        "std": None,
    }


def cvtColor(image):
    # 将 PIL 图像转成 RGB，主要服务于普通 jpg/png 流程。
    # 多光谱 tif 会在 dataloader 里以 numpy 数组读取，不应该走这里强制转 RGB。
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    image = image.convert("RGB")
    return image


def resize_image(image, size):
    # 原版推理用的 RGB letterbox resize。多光谱推理后面会单独补 numpy 版本。
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _reshape_band_vector(values, channels, name):
    # 将 [C] 形式的波段参数整理成 [1, 1, C]，方便和 HWC 影像逐通道广播。
    if values is None:
        return None

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != channels:
        raise ValueError(
            f"{name} 的长度必须等于当前输入通道数 {channels}，实际拿到 shape={arr.shape}。"
        )
    return arr.reshape((1, 1, channels))


def preprocess_input(image):
    # 普通 RGB 图像：
    #   原始值通常是 0-255，因此保持原版 HRNet 的 /255 逻辑。
    #
    # Sentinel-2 多光谱 tif：
    #   原始值通常是 uint16 反射率缩放值，例如 0-10000。这里先除以
    #   reflectance_scale，再按配置选择是否 clip 和 mean/std 标准化。
    image = image.astype(np.float32, copy=False)

    if image.size == 0:
        return image

    max_value = float(np.max(image))
    if max_value <= 255:
        image /= 255.0
        return image

    reflectance_scale = float(normalization_config.get("reflectance_scale", 10000.0))
    if reflectance_scale <= 0:
        raise ValueError(f"reflectance_scale 必须大于 0，当前值是 {reflectance_scale}。")
    image /= reflectance_scale

    channels = image.shape[2] if image.ndim == 3 else 1

    if normalization_config.get("enable_clip", False):
        clip_min = _reshape_band_vector(normalization_config.get("clip_min"), channels, "clip_min")
        clip_max = _reshape_band_vector(normalization_config.get("clip_max"), channels, "clip_max")
        if np.any(clip_max <= clip_min):
            raise ValueError("clip_max 的每个值都必须大于 clip_min 对应位置的值。")
        image = np.clip(image, clip_min, clip_max)

    if normalization_config.get("enable_mean_std", False):
        mean = _reshape_band_vector(normalization_config.get("mean"), channels, "mean")
        std = _reshape_band_vector(normalization_config.get("std"), channels, "std")
        if np.any(std <= 0):
            raise ValueError("std 的每个值都必须大于 0。")
        image = (image - mean) / std

    return image


def show_config(**kwargs):
    print("Configurations:")
    print("-" * 70)
    print("|%25s | %40s|" % ("keys", "values"))
    print("-" * 70)
    for key, value in kwargs.items():
        print("|%25s | %40s|" % (str(key), str(value)))
    print("-" * 70)


def download_weights(backbone, model_dir="./model_data"):
    import os

    from torch.hub import load_state_dict_from_url

    download_urls = {
        "hrnetv2_w18": "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w18_imagenet_pretrained.pth",
        "hrnetv2_w32": "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w32_imagenet_pretrained.pth",
        "hrnetv2_w48": "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w48_imagenet_pretrained.pth",
    }
    url = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
