#--------------------------------------------#
#   该脚本用于查看网络结构、参数量和 FLOPs。
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from multispectral_config import in_channels
from nets.hrnet import HRnet

if __name__ == "__main__":
    input_shape = [480, 480]
    # 当前多光谱光伏提取默认二分类：0=background，1=target。
    num_classes = 2
    backbone = "hrnetv2_w18"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # in_channels 来自 multispectral_config.py。
    # 统计 4/6 波段模型时，模型第一层和 dummy input 都必须使用相同通道数。
    model = HRnet(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False,
        in_channels=in_channels,
    ).to(device)

    summary(model, (in_channels, input_shape[0], input_shape[1]))

    dummy_input = torch.randn(1, in_channels, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(model.to(device), (dummy_input,), verbose=False)

    # thop 的 profile 默认通常只统计乘法。部分论文把卷积里的乘法和加法都记作
    # operations，因此这里沿用原项目习惯乘以 2。
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print("Input channels: %s" % in_channels)
    print("Total GFLOPS: %s" % flops)
    print("Total params: %s" % params)
