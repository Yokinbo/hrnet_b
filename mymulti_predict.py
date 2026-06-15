import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from hrnet import HRnet_Segmentation
from multispectral_config import (
    band_mode,
    image_ext,
    in_channels,
    selected_bands,
    trained_model_path,
    vis_bands,
)


# ===================== Editable config =====================
# Run directly:
#   python mymulti_predict.py
#
# input_path supports either one image or a folder.
EDITABLE_CONFIG = {
    "input_path": r"论文制图\测试图\原图",
    "label_dir": r"论文制图\测试图\label标签",
    "weights": trained_model_path,
    "output_dir": r"论文制图\6band测试结果",
    "device": "cuda:0",
    "input_size": 256,
    "suffixes": [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
    "save_mask": True,
    "save_overlay": True,
    "save_confusion": True,
    "label_suffixes": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
}
# ===========================================================


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def imwrite_unicode(path, image):
    """OpenCV on Windows may fail on Chinese paths; imencode+tofile is safer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise IOError(f"failed to encode image for: {path}")
    encoded.tofile(str(path))


def collect_input_images(input_path, suffixes):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        suffix_set = {suffix.lower() for suffix in suffixes}
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in suffix_set
        )
    raise FileNotFoundError(f"input path does not exist: {input_path}")


def find_label_for_image(image_path, label_dir, label_suffixes):
    if not label_dir:
        return None

    label_dir = Path(label_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f"label_dir does not exist: {label_dir}")

    for suffix in label_suffixes:
        label_path = label_dir / f"{image_path.stem}{suffix}"
        if label_path.exists():
            return label_path
    return None


def read_label_mask(label_path, target_shape):
    try:
        label = np.array(Image.open(label_path).convert("L"))
    except Exception as exc:
        raise FileNotFoundError(f"failed to read label: {label_path}") from exc

    if label.shape != target_shape:
        label = cv2.resize(
            label,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return (label > 0).astype(np.uint8)


def save_overlay(pred_mask, preview_img, output_overlay):
    preview = np.array(preview_img.convert("RGB"), dtype=np.uint8)
    red = np.zeros_like(preview)
    red[:, :, 0] = 255
    overlay = np.where(
        pred_mask[..., None] > 0,
        (0.55 * preview + 0.45 * red),
        preview,
    ).astype(np.uint8)
    imwrite_unicode(output_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_confusion_map(pred_mask, label_mask, output_confusion):
    """Save TN/FP/TP/FN map: TN black, FP blue, TP white, FN red."""
    pred01 = (pred_mask > 0).astype(np.uint8)
    label01 = (label_mask > 0).astype(np.uint8)

    rgb = np.zeros((label01.shape[0], label01.shape[1], 3), dtype=np.uint8)
    tp = (pred01 == 1) & (label01 == 1)
    fp = (pred01 == 1) & (label01 == 0)
    fn = (pred01 == 0) & (label01 == 1)

    rgb[tp] = [255, 255, 255]
    rgb[fp] = [0, 0, 255]
    rgb[fn] = [255, 0, 0]
    imwrite_unicode(output_confusion, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def predict_one_image(hrnet, image_path, args):
    t_start = time_synchronized()
    pred, preview_img, _, _ = hrnet.predict_mask(str(image_path))
    t_end = time_synchronized()

    pred_mask = (pred > 0).astype(np.uint8)
    output_dir = Path(args.output_dir)

    mask_path = output_dir / "模型预测mask_0-255" / f"{image_path.stem}_mask.png"
    overlay_path = output_dir / "红色半透明叠加图" / f"{image_path.stem}_overlay.png"
    label_view_path = output_dir / "人工标签可视化0-255" / f"{image_path.stem}_label.png"
    confusion_path = output_dir / "TP_FP_FN_TN彩色误差图" / f"{image_path.stem}_confusion.png"

    if args.save_mask:
        imwrite_unicode(mask_path, pred_mask * 255)

    if args.save_overlay:
        save_overlay(pred_mask, preview_img, overlay_path)

    if args.save_confusion:
        label_path = find_label_for_image(image_path, args.label_dir, args.label_suffixes)
        if label_path is None:
            print(f"[warn] no label found for {image_path.name}, skip confusion map")
        else:
            label_mask = read_label_mask(label_path, pred_mask.shape)
            imwrite_unicode(label_view_path, label_mask * 255)
            save_confusion_map(pred_mask, label_mask, confusion_path)

    print(f"[done] {image_path.name} inference={t_end - t_start:.4f}s")


def main(args):
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file does not exist: {weights_path}")

    image_paths = collect_input_images(args.input_path, args.suffixes)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {args.input_path}")

    use_cuda = args.device.lower().startswith("cuda") and torch.cuda.is_available()

    print("Current HRNet multispectral prediction config:")
    print(f"  band_mode     : {band_mode}")
    print(f"  image_ext     : {image_ext}")
    print(f"  selected_bands: {selected_bands}")
    print(f"  vis_bands     : {vis_bands}")
    print(f"  in_channels   : {in_channels}")
    print(f"  weights       : {args.weights}")
    print(f"  input_path    : {args.input_path}")
    print(f"  label_dir     : {args.label_dir or '(disabled)'}")
    print(f"  image_count   : {len(image_paths)}")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  device        : {'cuda' if use_cuda else 'cpu'}")

    hrnet = HRnet_Segmentation(
        model_path=args.weights,
        num_classes=2,
        image_ext=image_ext,
        selected_bands=selected_bands,
        vis_bands=vis_bands,
        in_channels=in_channels,
        input_shape=[args.input_size, args.input_size],
        mix_type=0,
        cuda=use_cuda,
    )

    for image_path in image_paths:
        predict_one_image(hrnet, image_path, args)

    print("Saved outputs to:", args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="HRNet multispectral batch prediction")
    parser.add_argument("--input-path", default=EDITABLE_CONFIG["input_path"], help="input image file or folder")
    parser.add_argument("--label-dir", default=EDITABLE_CONFIG["label_dir"], help="manual label folder")
    parser.add_argument("--weights", default=EDITABLE_CONFIG["weights"], help="model weights path")
    parser.add_argument("--output-dir", default=EDITABLE_CONFIG["output_dir"], help="output folder")
    parser.add_argument("--device", default=EDITABLE_CONFIG["device"], help="prediction device, e.g. cuda:0 or cpu")
    parser.add_argument("--input-size", default=EDITABLE_CONFIG["input_size"], type=int, help="square inference input size")
    parser.add_argument("--suffixes", nargs="+", default=EDITABLE_CONFIG["suffixes"], help="image suffixes for folder input")
    parser.add_argument("--label-suffixes", nargs="+", default=EDITABLE_CONFIG["label_suffixes"], help="label suffixes")
    parser.add_argument("--save-mask", action="store_true", default=EDITABLE_CONFIG["save_mask"])
    parser.add_argument("--no-save-mask", action="store_false", dest="save_mask")
    parser.add_argument("--save-overlay", action="store_true", default=EDITABLE_CONFIG["save_overlay"])
    parser.add_argument("--no-save-overlay", action="store_false", dest="save_overlay")
    parser.add_argument("--save-confusion", action="store_true", default=EDITABLE_CONFIG["save_confusion"])
    parser.add_argument("--no-save-confusion", action="store_false", dest="save_confusion")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
