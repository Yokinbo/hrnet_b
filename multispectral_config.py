"""
多光谱训练、验证、推理的统一配置文件。

后续代码会统一从这里读取影像后缀、当前使用的波段组合、输入通道数、
可视化波段以及归一化参数。这样切换 3 / 4 / 6 波段实验时，只需要优先
修改这个文件，避免 train.py、hrnet.py、get_miou.py 里到处手动同步。
"""

# -------------------------------------------------------------------
# 推理阶段默认权重
# -------------------------------------------------------------------
# 这个路径主要供 hrnet.py / predict.py / get_miou.py 使用。
# 训练阶段是否加载预训练权重，仍然以 train.py 里的 model_path 为准。
trained_model_path = "logs/6band/best_epoch_weights.pth"


# -------------------------------------------------------------------
# 数据影像格式
# -------------------------------------------------------------------
# 原始 HRNet 代码默认读取 .jpg；多光谱训练时，VOCdevkit/VOC2007/JPEGImages
# 目录里通常放的是多波段 .tif patch。
image_ext = ".tif"


# -------------------------------------------------------------------
# 波段模式
# -------------------------------------------------------------------
# 可选值：
# - "rgb"   : 使用真彩色三波段 [B4, B3, B2]
# - "4band" : 使用 [B2, B3, B4, B8]
# - "6band" : 使用 [B2, B3, B4, B8, B11, B12]
#
# 注意：
# 这里的波段编号是从 1 开始写的，符合遥感软件里常见的“第 1 波段、第 2 波段”
# 说法；实际读取 numpy 数组时，代码会再转换成从 0 开始的索引。
band_mode = "6band"


# -------------------------------------------------------------------
# 原始 tif 的波段顺序约定
# -------------------------------------------------------------------
# 当前假定你的 6 波段 tif 顺序固定为：
# [1, 2, 3, 4, 5, 6] = [B2, B3, B4, B8, B11, B12]
#
# 如果你后面导出的 tif 波段顺序不同，只需要改这里的映射关系。
band_options = {
    "rgb": [3, 2, 1],
    "4band": [1, 2, 3, 4],
    "6band": [1, 2, 3, 4, 5, 6],
}

selected_bands = band_options[band_mode]

# 模型第一层卷积需要知道输入通道数。
# 比如 band_mode="6band" 时，这里就是 6。
in_channels = len(selected_bands)


# -------------------------------------------------------------------
# 可视化波段
# -------------------------------------------------------------------
# 多光谱输入本身可能有 4 或 6 个通道，但预测结果叠加显示时仍然需要一张
# 人眼可读的 RGB 底图。这里默认使用真彩色 [B4, B3, B2]。
vis_bands = [3, 2, 1]


# -------------------------------------------------------------------
# 多光谱归一化配置
# -------------------------------------------------------------------
# 设计思路：
# 1. Sentinel-2 常见 uint16 反射率数据通常需要先除以 10000。
# 2. clip_min / clip_max 用于抑制异常极值，数值应在除以 10000 之后的尺度上。
# 3. mean / std 用于按通道标准化，最好来自训练集统计值。
#
# 下面这组参数沿用 UNet_b 当前多光谱实验配置，后续如果你换数据集，建议重新
# 统计训练集再更新这些值。
normalization_configs = {
    "rgb": {
        # selected_bands = [3, 2, 1]，对应 [B4, B3, B2]
        "reflectance_scale": 10000.0,
        "enable_clip": True,
        "clip_min": [0.037500, 0.051400, 0.029200],
        "clip_max": [0.323400, 0.250800, 0.195000],
        "enable_mean_std": True,
        "mean": [0.183408, 0.143386, 0.101527],
        "std": [0.066324, 0.047525, 0.035681],
    },
    "4band": {
        # selected_bands = [1, 2, 3, 4]，对应 [B2, B3, B4, B8]
        "reflectance_scale": 10000.0,
        "enable_clip": True,
        "clip_min": [0.029200, 0.051400, 0.037500, 0.094900],
        "clip_max": [0.195000, 0.250800, 0.323400, 0.408400],
        "enable_mean_std": True,
        "mean": [0.101527, 0.143386, 0.183408, 0.259956],
        "std": [0.035681, 0.047525, 0.066324, 0.070516],
    },
    "6band": {
        # selected_bands = [1, 2, 3, 4, 5, 6]
        # 对应 [B2, B3, B4, B8, B11, B12]
        "reflectance_scale": 10000.0,
        "enable_clip": True,
        "clip_min": [0.029200, 0.051400, 0.037500, 0.094900, 0.146000, 0.082400],
        "clip_max": [0.195000, 0.250800, 0.323400, 0.408400, 0.469300, 0.455800],
        "enable_mean_std": True,
        "mean": [0.101527, 0.143386, 0.183408, 0.259956, 0.341008, 0.294076],
        "std": [0.035681, 0.047525, 0.066324, 0.070516, 0.071065, 0.079996],
    },
}

# 当前波段模式对应的归一化参数。
# 后续 preprocess_input 会读取它，保证训练、验证、推理使用同一套预处理。
normalization_config = normalization_configs[band_mode]


# -------------------------------------------------------------------
# Training-time online data augmentation
# -------------------------------------------------------------------
# These augmentations are used only for the training set. Validation and test
# samples remain unchanged, so different models can be compared under the same
# data distribution.
train_augmentation_config = {
    "enabled": True,

    # Multispectral reflectance perturbation.
    # Simulates seasonal, solar-angle, atmospheric and surface-moisture changes.
    "reflectance_prob": 0.50,
    "reflectance_global_range": [0.90, 1.10],
    "reflectance_band_range": [0.95, 1.05],

    # Geometry perturbation.
    # Simulates different PV array directions and cutting orientations.
    "geometry_prob": 0.50,

    # Soft local shadow / thin cloud-shadow perturbation.
    # Simulates residual cloud shadow, terrain shadow and PV array shadow.
    "shadow_prob": 0.25,
    "shadow_factor_range": [0.75, 0.90],
    "shadow_radius_range": [0.25, 0.45],

    # Mild Gaussian noise.
    # Simulates sensor noise, atmospheric residuals and local abnormal pixels.
    "noise_prob": 0.25,
    "noise_sigma_range": [0.003, 0.008],

    # Random scale by crop and resize back.
    # Simulates different PV plant sizes and patch cutting positions.
    "scale_prob": 0.20,
    "scale_crop_range": [0.85, 1.00],
}
