import os

from tqdm import tqdm

from hrnet import HRnet_Segmentation
from multispectral_config import image_ext, in_channels, selected_bands, trained_model_path
from utils.utils_metrics import compute_mIoU, show_results

"""
独立 mIoU 评估脚本。

当前版本适配多光谱 HRNet：
1. 数据仍使用 VOCdevkit/VOC2007 目录结构。
2. 影像后缀、波段组合、输入通道数、默认权重路径统一来自 multispectral_config.py。
3. 预测时直接把影像路径交给 HRnet_Segmentation，底层会自动处理 tif 多波段读取。
"""

if __name__ == "__main__":
    # miou_mode = 0：完整流程，先生成预测 mask，再计算 mIoU。
    # miou_mode = 1：只生成预测 mask。
    # miou_mode = 2：只计算 mIoU，需要 miou_out/detection-results 已经存在。
    miou_mode = 0

    # 光伏提取默认二分类：0=background，1=target。
    num_classes = 2
    name_classes = ["background", "pv"]

    VOCdevkit_path = "VOCdevkit"

    # 使用哪个划分文件做评估。开发调参通常用 val.txt；最终报告可改成 test.txt。
    image_set = "val.txt"
    image_ids = open(
        os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation", image_set),
        "r",
        encoding="utf-8",
    ).read().splitlines()

    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, "detection-results")

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        hrnet = HRnet_Segmentation(
            model_path=trained_model_path,
            num_classes=num_classes,
            image_ext=image_ext,
            selected_bands=selected_bands,
            in_channels=in_channels,
            input_shape=[256, 256],
        )
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            # 多光谱 tif 直接传路径。hrnet.py 会用 rasterio 按 selected_bands 读取。
            image_path = os.path.join(
                VOCdevkit_path,
                "VOC2007/JPEGImages",
                image_id + image_ext,
            )
            image = hrnet.get_miou_png(image_path)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir,
            pred_dir,
            image_ids,
            num_classes,
            name_classes,
        )
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
