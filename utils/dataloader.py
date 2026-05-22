import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input

try:
    from multispectral_config import normalization_config
except Exception:
    normalization_config = {"reflectance_scale": 10000.0}


class SegmentationDataset(Dataset):
    def __init__(
        self,
        annotation_lines,
        input_shape,
        num_classes,
        train,
        dataset_path,
        image_ext=".tif",
        selected_bands=None,
        augmentation_config=None,
    ):
        super(SegmentationDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

        # 多光谱改造新增：
        # image_ext 控制从 JPEGImages 目录读取 .jpg 还是 .tif；
        # selected_bands 控制多波段 tif 实际取哪些波段，使用 1-based 编号。
        self.image_ext = image_ext
        self.selected_bands = selected_bands
        self.augmentation_config = augmentation_config or {"enabled": False}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        image_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + self.image_ext)
        label_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".png")

        if self.image_ext.lower() in [".tif", ".tiff"]:
            image = self.read_tif(image_path)
        else:
            image = Image.open(image_path)
        label = Image.open(label_path)

        image, label = self.get_random_data(image, label, self.input_shape, random=self.train)

        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        label = np.array(label)

        if self._use_training_augmentation(image):
            image, label = self._augment_training_sample(image, label)

        image = np.transpose(preprocess_input(image), [2, 0, 1])
        label[label >= self.num_classes] = self.num_classes

        # 转成 one-hot。多出来的最后一类用于 ignore_index，和原版 HRNet 保持一致。
        seg_labels = np.eye(self.num_classes + 1)[label.reshape([-1])]
        seg_labels = seg_labels.reshape(
            (int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1)
        )

        return image, label, seg_labels

    def read_tif(self, image_path):
        # rasterio 按 (C, H, W) 读取多波段 tif；网络和增强更习惯 HWC，
        # 所以这里统一转成 (H, W, C)。
        #
        # selected_bands 使用从 1 开始的波段编号，例如 [3, 2, 1] 表示
        # 从 6 波段 Sentinel-2 tif 中读取 [B4, B3, B2]。
        import rasterio

        with rasterio.open(image_path) as src:
            if self.selected_bands is None:
                band_indexes = list(range(1, src.count + 1))
            else:
                band_indexes = self.selected_bands

            image = src.read(indexes=band_indexes)
            image = np.transpose(image, (1, 2, 0))
        return image

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def _use_training_augmentation(self, image):
        return (
            self.train
            and self.augmentation_config.get("enabled", False)
            and self.image_ext.lower() in [".tif", ".tiff"]
            and isinstance(image, np.ndarray)
            and image.ndim == 3
        )

    def _to_reflectance(self, image):
        scale = float(normalization_config.get("reflectance_scale", 10000.0))
        if scale <= 0:
            raise ValueError("normalization_config['reflectance_scale'] must be greater than 0.")
        return image.astype(np.float32, copy=True) / scale

    def _from_reflectance(self, reflectance):
        scale = float(normalization_config.get("reflectance_scale", 10000.0))
        return (reflectance.astype(np.float32, copy=False) * scale).astype(np.float32)

    def _augment_training_sample(self, image, label):
        cfg = self.augmentation_config
        reflectance = self._to_reflectance(image)
        label = label.astype(np.uint8, copy=False)

        if random.random() < cfg.get("geometry_prob", 0.0):
            reflectance, label = self._augment_geometry(reflectance, label)

        if random.random() < cfg.get("scale_prob", 0.0):
            reflectance, label = self._augment_random_scale(reflectance, label)

        if random.random() < cfg.get("reflectance_prob", 0.0):
            reflectance = self._augment_reflectance(reflectance)

        if random.random() < cfg.get("shadow_prob", 0.0):
            reflectance = self._augment_shadow(reflectance)

        if random.random() < cfg.get("noise_prob", 0.0):
            reflectance = self._augment_noise(reflectance)

        return self._from_reflectance(reflectance), label

    def _augment_geometry(self, image, label):
        op = random.choice(["hflip", "vflip", "rot90", "rot180", "rot270"])
        if op == "hflip":
            return np.ascontiguousarray(image[:, ::-1, :]), np.ascontiguousarray(label[:, ::-1])
        if op == "vflip":
            return np.ascontiguousarray(image[::-1, :, :]), np.ascontiguousarray(label[::-1, :])

        original_h, original_w = image.shape[:2]
        k = {"rot90": 1, "rot180": 2, "rot270": 3}[op]
        image = np.rot90(image, k=k).copy()
        label = np.rot90(label, k=k).copy()
        if image.shape[:2] != (original_h, original_w):
            image = cv2.resize(image, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            if image.ndim == 2:
                image = image[:, :, None]
        return image.astype(np.float32), label.astype(np.uint8)

    def _augment_reflectance(self, image):
        cfg = self.augmentation_config
        global_low, global_high = cfg.get("reflectance_global_range", [0.90, 1.10])
        band_low, band_high = cfg.get("reflectance_band_range", [0.95, 1.05])
        global_factor = random.uniform(global_low, global_high)
        band_factors = np.random.uniform(
            band_low,
            band_high,
            size=(1, 1, image.shape[2]),
        ).astype(np.float32)
        return image * global_factor * band_factors

    def _augment_shadow(self, image):
        cfg = self.augmentation_config
        factor_low, factor_high = cfg.get("shadow_factor_range", [0.75, 0.90])
        radius_low, radius_high = cfg.get("shadow_radius_range", [0.25, 0.45])
        h, w = image.shape[:2]
        center_y = random.uniform(-0.5, 0.5)
        center_x = random.uniform(-0.5, 0.5)
        radius = random.uniform(radius_low, radius_high)
        yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        shadow = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / max(radius, 1e-6))
        factor = random.uniform(factor_low, factor_high)
        shadow_map = 1.0 - (1.0 - factor) * shadow
        return image * shadow_map[:, :, None]

    def _augment_noise(self, image):
        sigma_low, sigma_high = self.augmentation_config.get("noise_sigma_range", [0.003, 0.008])
        sigma = random.uniform(sigma_low, sigma_high)
        noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
        return image + noise

    def _augment_random_scale(self, image, label):
        crop_low, crop_high = self.augmentation_config.get("scale_crop_range", [0.85, 1.00])
        ratio = random.uniform(crop_low, crop_high)
        h, w = image.shape[:2]
        crop_h = max(8, int(h * ratio))
        crop_w = max(8, int(w * ratio))
        top = random.randint(0, max(0, h - crop_h))
        left = random.randint(0, max(0, w - crop_w))

        image_crop = image[top:top + crop_h, left:left + crop_w, :]
        label_crop = label[top:top + crop_h, left:left + crop_w]
        image = cv2.resize(image_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)
        if image.ndim == 2:
            image = image[:, :, None]
        return image.astype(np.float32), label.astype(np.uint8)

    def _image_size(self, image):
        if isinstance(image, Image.Image):
            return image.size
        height, width = image.shape[:2]
        return width, height

    def _resize_array(self, image, size, interpolation=cv2.INTER_LINEAR):
        resized = cv2.resize(image, size, interpolation=interpolation)
        if resized.ndim == 2:
            resized = np.expand_dims(resized, -1)
        return resized

    def _paste_array(self, image, canvas_shape, dx, dy, fill_value=128):
        # PIL paste 会自动裁掉越界部分；numpy 需要手动求源图和画布的交集。
        h, w, channels = canvas_shape
        src_h, src_w = image.shape[:2]
        canvas = np.full((h, w, channels), fill_value, dtype=image.dtype)

        x1 = max(dx, 0)
        y1 = max(dy, 0)
        x2 = min(dx + src_w, w)
        y2 = min(dy + src_h, h)

        src_x1 = max(-dx, 0)
        src_y1 = max(-dy, 0)
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)

        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2, :] = image[src_y1:src_y2, src_x1:src_x2, :]
        return canvas

    def _warp_array(self, image, matrix, size, flags, border_value=128):
        # OpenCV 对 5/6 通道数组的 borderValue 支持不稳定，因此多通道时逐通道旋转。
        channels = image.shape[2]
        warped_channels = []
        for c in range(channels):
            warped = cv2.warpAffine(
                image[:, :, c],
                matrix,
                size,
                flags=flags,
                borderValue=border_value,
            )
            warped_channels.append(warped)
        return np.stack(warped_channels, axis=-1)

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        # 普通图片仍走 PIL RGB 流程；多光谱 tif 是 numpy 数组，不能强制 convert("RGB")。
        if isinstance(image, Image.Image):
            image = cvtColor(image)
        else:
            image = np.asarray(image)
            if image.ndim == 2:
                image = np.expand_dims(image, -1)

        label = Image.fromarray(np.array(label))
        iw, ih = self._image_size(image)
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            if isinstance(image, Image.Image):
                image = image.resize((nw, nh), Image.BICUBIC)
                new_image = Image.new("RGB", [w, h], (128, 128, 128))
                new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            else:
                image = self._resize_array(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
                new_image = self._paste_array(
                    image,
                    (h, w, image.shape[2]),
                    (w - nw) // 2,
                    (h - nh) // 2,
                )

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new("L", [w, h], 0)
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        # 随机缩放与长宽扰动，保持原版数据增强思路。
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        if isinstance(image, Image.Image):
            image = image.resize((nw, nh), Image.BICUBIC)
        else:
            image = self._resize_array(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        label = label.resize((nw, nh), Image.NEAREST)

        flip = self.rand() < 0.5
        if flip:
            if isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                image = np.ascontiguousarray(image[:, ::-1, :])
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        if isinstance(image, Image.Image):
            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image = new_image
        else:
            image = self._paste_array(image, (h, w, image.shape[2]), dx, dy)

        new_label = Image.new("L", (w, h), 0)
        new_label.paste(label, (dx, dy))
        label = new_label

        image_data = np.array(image)

        # 高斯模糊只对普通 RGB 图像保留。多光谱各波段有物理含义，先不做颜色/模糊类增强。
        if image_data.ndim == 3 and image_data.shape[2] == 3:
            blur = self.rand() < 0.25
            if blur:
                image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            matrix = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            if image_data.ndim == 3:
                image_data = self._warp_array(
                    image_data,
                    matrix,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    border_value=128,
                )
            else:
                image_data = cv2.warpAffine(
                    image_data,
                    matrix,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderValue=128,
                )
            label = cv2.warpAffine(
                np.array(label, np.uint8),
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )

        # HSV 色彩增强只适合普通 RGB 图像；4/6 波段多光谱不做这一步。
        if image_data.ndim == 3 and image_data.shape[2] == 3:
            r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
            hue_img, sat_img, val_img = cv2.split(cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_RGB2HSV))
            dtype = hue_img.dtype

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            image_data = cv2.merge(
                (
                    cv2.LUT(hue_img, lut_hue),
                    cv2.LUT(sat_img, lut_sat),
                    cv2.LUT(val_img, lut_val),
                )
            )
            image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


def seg_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
