import numpy as np
import cv2
from pathlib import Path
from skimage.metrics import structural_similarity as ssim


# 如果不想装 skimage，PSNR 可以手写，但 SSIM 强烈建议用库

def calculate_metrics(img_original, img_denoised):
    """
    计算图像质量指标 PSNR 和 SSIM

    参数:
    - img_original: 原始清晰图像 (numpy array)
    - img_denoised: 去噪后的图像 (numpy array)

    返回:
    - 一个字典，包含 'PSNR' 和 'SSIM' 的值
    """

    # 1. 安全检查：确保尺寸一致
    if img_original.shape != img_denoised.shape:
        raise ValueError(f"尺寸不匹配! 原图: {img_original.shape}, 处理图: {img_denoised.shape}")

    # -------------------------------
    # 计算 PSNR (峰值信噪比)
    # -------------------------------
    # PSNR = 10 * log10(MAX^2 / MSE)
    # MAX 通常是 255

    img1 = img_original.astype(np.float64)
    img2 = img_denoised.astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        # 如果 MSE 为 0，说明两张图一模一样，PSNR 趋近无限大
        psnr_val = float('inf')
    else:
        max_pixel = 255.0
        psnr_val = 10 * np.log10((max_pixel ** 2) / mse)

    # -------------------------------
    # 计算 SSIM (结构相似性)
    # -------------------------------
    # data_range=255 告诉函数像素范围是 0-255
    # 如果是灰度图，通常不需要指定 channel_axis，或者是 None
    # win_size 默认是 7，可以不填

    ssim_val = ssim(img_original, img_denoised, data_range=255)

    return {
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "MSE": mse  # 顺便返回均方误差
    }
