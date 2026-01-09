import torch
import pyiqa
import cv2
import numpy as np


# 这是一个强大的库，会自动下载模型权重
# 首次运行需要联网下载 VGG 模型 (约几百 MB)

def compare_advanced_metrics(img_original_path, img_denoised_path):
    """
    计算 LPIPS (有参考) 和 NIQE (无参考)
    """
    # 1. 初始化评估器 (使用 GPU 如果有的话，否则 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # LPIPS: 越低越好 (感知差异)
    lpips_metric = pyiqa.create_metric('lpips', device=device)

    # NIQE: 越低越好 (自然度)
    niqe_metric = pyiqa.create_metric('niqe', device=device)

    # 2. 读取图片并转为 Tensor (归一化到 0-1)
    # pyiqa 这里的读取会自动处理 batch 维度
    # 我们用 OpenCV 读，然后手动转一下格式比较保险，或者直接传路径

    # 方式 A: 直接传路径 (pyiqa 支持)
    # score = metric(path1, path2)

    print(f"正在计算指标: {img_denoised_path} ...")

    # LPIPS 需要两张图对比
    lpips_score = lpips_metric(img_original_path, img_denoised_path).item()

    # NIQE 只需要这一张图
    niqe_score = niqe_metric(img_denoised_path).item()

    return lpips_score, niqe_score


# ==========================================
# 模拟你的测试场景
# ==========================================

if __name__ == "__main__":
    # 假设你已经生成了这些图片 (用之前的代码)
    path_original = "../../data/classic_photo/lena_gray.png"

    # 假设这是刚才生成的两张结果图的路径
    path_gaussian = "../../out/baseline/gaussian_filter/lena_gray_gaussian_sigma1.0.png"
    # 注意：你需要先运行之前的双边滤波代码生成这张图
    path_bilateral = "../../out/baseline/neighborhood_filter/lena_gray_bilateral_h30_rho5_s5.png"

    # 检查文件是否存在
    import os

    if not os.path.exists(path_bilateral):
        print("请先运行双边滤波代码生成图片！")
    else:
        print(f"{'算法':<15} | {'LPIPS (低好)':<15} | {'NIQE (低好)':<15}")
        print("-" * 50)

        # 1. 测高斯
        l_gauss, n_gauss = compare_advanced_metrics(path_original, path_gaussian)
        print(f"{'Gaussian':<15} | {l_gauss:.4f}          | {n_gauss:.4f}")

        # 2. 测双边
        l_bi, n_bi = compare_advanced_metrics(path_original, path_bilateral)
        print(f"{'Bilateral':<15} | {l_bi:.4f}          | {n_bi:.4f}")

        print("-" * 50)

        if l_bi < l_gauss:
            print("结论：LPIPS 认为双边滤波更接近人眼视觉！(因为它更低)")
        else:
            print("结论：LPIPS 认为两者差不多，或者参数还需要调优。")