import colorsys
import copy
import time

import cv2
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from multispectral_config import image_ext, in_channels, selected_bands, trained_model_path, vis_bands
from nets.hrnet import HRnet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


class HRnet_Segmentation(object):
    _defaults = {
        # 推理阶段默认使用的权重路径统一从 multispectral_config.py 读取。
        # 训练阶段是否加载权重仍由 train.py 里的 model_path 控制。
        "model_path": trained_model_path,
        # 光伏提取默认二分类：0=background，1=target。
        "num_classes": 2,
        "backbone": "hrnetv2_w18",
        # 多光谱推理配置。selected_bands 使用 1-based 波段编号。
        "image_ext": image_ext,
        "selected_bands": selected_bands,
        "vis_bands": vis_bands,
        "in_channels": in_channels,
        "input_shape": [480, 480],
        # mix_type = 0：原图和分割结果混合
        # mix_type = 1：只显示分割彩图
        # mix_type = 2：只保留原图中的目标区域
        "mix_type": 0,
        "cuda": True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 如果用户只改 selected_bands，没有手动同步 in_channels，这里自动修正。
        self.in_channels = len(self.selected_bands)

        if self.num_classes <= 21:
            self.colors = [
                (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
                (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                (0, 64, 128), (128, 64, 12),
            ]
        else:
            hsv_tuples = [(x / self.num_classes, 1.0, 1.0) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors)
            )

        self.generate()
        show_config(
            model_path=self.model_path,
            num_classes=self.num_classes,
            backbone=self.backbone,
            image_ext=self.image_ext,
            selected_bands=self.selected_bands,
            vis_bands=self.vis_bands,
            in_channels=self.in_channels,
            input_shape=self.input_shape,
            mix_type=self.mix_type,
            cuda=self.cuda,
        )

    def generate(self, onnx=False):
        # 推理时构建模型也传入 in_channels，必须和训练时的波段数一致。
        self.net = HRnet(
            num_classes=self.num_classes,
            backbone=self.backbone,
            pretrained=False,
            in_channels=self.in_channels,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print("{} model, and classes loaded.".format(self.model_path))
        if not onnx and self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def read_inference_image(self, image):
        # 推理输入支持：
        # 1. 文件路径：.tif/.tiff 用 rasterio 读取，多波段按 selected_bands 选取
        # 2. PIL.Image：普通 RGB 图片
        # 3. numpy.ndarray：已经在外部读好的多通道影像
        if isinstance(image, str):
            if image.lower().endswith((".tif", ".tiff")):
                with rasterio.open(image) as src:
                    arr = src.read(indexes=self.selected_bands)  # (C, H, W)
                    arr = np.transpose(arr, (1, 2, 0))           # -> (H, W, C)
                return arr
            return Image.open(image)
        return image

    def resize_multiband_image(self, image, size):
        # 多波段 numpy 影像的 letterbox resize，逻辑对应 utils.resize_image 的 PIL 版本。
        ih, iw = image.shape[:2]
        w, h = size

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if image.ndim == 2:
            image = np.expand_dims(image, -1)

        channels = image.shape[2]
        new_image = np.zeros((h, w, channels), dtype=image.dtype)
        new_image[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
        return new_image, nw, nh

    def make_vis_image(self, image):
        # 多光谱输入本身不能直接显示。这里按 vis_bands 取真彩色波段，
        # 并做 2%-98% 分位拉伸，生成用于叠加分割结果的 RGB 预览图。
        if isinstance(image, Image.Image):
            return copy.deepcopy(image)

        image = np.asarray(image)
        # image 已经按 selected_bands 抽过波段，所以 vis_bands 需要先从
        # “原始 tif 波段编号”映射到“当前数组通道位置”。
        band_to_channel = {band: i for i, band in enumerate(self.selected_bands)}
        idx = [band_to_channel[b] for b in self.vis_bands if b in band_to_channel]
        if len(idx) != 3:
            raise ValueError(
                f"vis_bands={self.vis_bands} 必须都包含在 selected_bands={self.selected_bands} 中，"
                "否则无法生成 RGB 预览图。"
            )
        rgb = image[:, :, idx].astype(np.float32)
        out = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(rgb.shape[2]):
            band = rgb[:, :, i]
            low = np.percentile(band, 2)
            high = np.percentile(band, 98)
            if high <= low:
                scaled = np.zeros_like(band, dtype=np.uint8)
            else:
                scaled = ((band - low) / (high - low) * 255.0).clip(0, 255).astype(np.uint8)
            out[:, :, i] = scaled
        return Image.fromarray(out)

    def prepare_input(self, image):
        # 所有推理方法共用同一套输入准备逻辑，避免 detect_image / FPS / mIoU
        # 各自维护一份预处理代码后出现不一致。
        image = self.read_inference_image(image)

        if isinstance(image, Image.Image):
            image = cvtColor(image)
            old_img = copy.deepcopy(image)
            original_h = np.array(image).shape[0]
            original_w = np.array(image).shape[1]
            image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
            image_data = np.expand_dims(
                np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
                0,
            )
        else:
            image = np.array(image, np.float32)
            if image.ndim == 2:
                image = np.expand_dims(image, -1)
            old_img = self.make_vis_image(image)
            original_h, original_w = image.shape[0], image.shape[1]
            image_data, nw, nh = self.resize_multiband_image(
                image,
                (self.input_shape[1], self.input_shape[0]),
            )
            image_data = np.expand_dims(
                np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
                0,
            )

        return old_img, image_data, nw, nh, original_h, original_w

    def predict_mask(self, image):
        old_img, image_data, nw, nh, original_h, original_w = self.prepare_input(image)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[
                int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw),
            ]
            pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        return pr, old_img, original_h, original_w

    def detect_image(self, image, count=False, name_classes=None):
        pr, old_img, original_h, original_w = self.predict_mask(image)

        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = original_h * original_w
            print("-" * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print("-" * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    class_name = str(name_classes[i]) if name_classes is not None else str(i)
                    print("|%25s | %15s | %14.2f%%|" % (class_name, str(num), ratio))
                    print("-" * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            seg_img = np.reshape(
                np.array(self.colors, np.uint8)[np.reshape(pr, [-1])],
                [original_h, original_w, -1],
            )
            image = Image.fromarray(np.uint8(seg_img))
            image = Image.blend(old_img, image, 0.7)
        elif self.mix_type == 1:
            seg_img = np.reshape(
                np.array(self.colors, np.uint8)[np.reshape(pr, [-1])],
                [original_h, original_w, -1],
            )
            image = Image.fromarray(np.uint8(seg_img))
        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype("uint8")
            image = Image.fromarray(np.uint8(seg_img))
        else:
            raise ValueError(f"Unsupported mix_type: {self.mix_type}")

        return image

    def get_FPS(self, image, test_interval):
        _, image_data, nw, nh, _, _ = self.prepare_input(image)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            pr = pr[
                int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw),
            ]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                pr = pr[
                    int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw),
                ]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx

        self.generate(onnx=True)

        # ONNX 假输入也要使用当前多光谱输入通道数。
        im = torch.zeros(1, self.in_channels, *self.input_shape).to("cpu")
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        print(f"Starting export with onnx {onnx.__version__}.")
        torch.onnx.export(
            self.net,
            im,
            f=model_path,
            verbose=False,
            opset_version=12,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=input_layer_names,
            output_names=output_layer_names,
            dynamic_axes=None,
        )

        model_onnx = onnx.load(model_path)
        onnx.checker.check_model(model_onnx)

        if simplify:
            import onnxsim

            print(f"Simplifying with onnx-simplifier {onnxsim.__version__}.")
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None,
            )
            assert check, "assert check failed"
            onnx.save(model_onnx, model_path)

        print("Onnx model save as {}".format(model_path))

    def get_miou_png(self, image):
        pr, _, _, _ = self.predict_mask(image)
        image = Image.fromarray(np.uint8(pr))
        return image
