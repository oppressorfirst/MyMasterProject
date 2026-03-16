import os
import cv2
import numpy as np
import bm3d
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import matplotlib.pyplot as plt
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import njit, prange
from tqdm import tqdm, trange
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import csv
from numba import njit, prange
import time
from scipy.fft import dctn, idctn  # 引入 3D 变换库
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import concurrent.futures

# -------------------------
# 读取 PNG 并返回 YUV
# -------------------------
def read_png_to_yuv(path, normalize=True):
    img = cv2.imread(path)
    if img is None:
        return None, None, None, None

    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = yuv[:, :, 0]
    cr = yuv[:, :, 1]  # 注意：OpenCV 中索引1是 Cr
    cb = yuv[:, :, 2]  # 索引2是 Cb

    if normalize:
        y = y.astype(np.float32) / 255.0
        cb = cb.astype(np.float32) / 255.0
        cr = cr.astype(np.float32) / 255.0

    return y, cb, cr, img


# -------------------------
# 添加噪声
# -------------------------
def add_awgn_noise(y, sigma=25, seed=42):
    rng = np.random.default_rng(seed)
    sigma_norm = sigma / 255.0
    noise = rng.normal(0, sigma_norm, y.shape)
    noisy_y = np.clip(y + noise, 0, 1)
    return noisy_y


def add_poisson_gaussian_noise(img_clean, a=0.1, sigma_norm=25/255, seed=None):
    """
    为 [0, 1] 范围的图像添加真实的泊松-高斯混合噪声 (支持复现)。
    参数:
        img_clean: 干净的原图 (float32 or float64, 范围 0~1)
        a: 泊松增益 (Photon Gain)。常用测试范围 0.005 ~ 0.05
        b: 高斯读取噪声方差 (Read Noise Variance)。常用测试范围 0.0001 ~ 0.005
        seed: 随机数种子。传入一个整数(如 42)即可保证每次生成的噪声完全一致。
    """
    # 使用局部随机生成器，不会影响外部代码 (如 AKNN) 的 np.random 状态
    rng = np.random.default_rng(seed)

    # 1. 模拟泊松噪声
    photon_counts = np.maximum(img_clean / a, 1e-10)
    noisy_poisson = rng.poisson(photon_counts) * a

    # 2. 模拟高斯噪声
    noisy_gaussian = rng.normal(0, sigma_norm, img_clean.shape)

    # 3. 混合并限制范围
    noisy_img = noisy_poisson + noisy_gaussian

    return np.clip(noisy_img, 0.0, 1.0).astype(np.float32)

# -------------------------
# 可视化并保存
# -------------------------
def showPic(img_bgr, y, y_noise, cb, cr, y_denoised, img_save_dir, idx):
    def to_bgr_uint8(img_gray):
        gray_255 = (np.clip(img_gray, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(gray_255, cv2.COLOR_GRAY2BGR)

    h, w, _ = img_bgr.shape

    # 合成彩色图 (顺序必须是 Y, Cr, Cb)
    noisy_yuv = np.stack([(y_noise * 255).astype(np.uint8),
                          (cr * 255).astype(np.uint8),
                          (cb * 255).astype(np.uint8)], axis=2)
    noisy_bgr = cv2.cvtColor(noisy_yuv, cv2.COLOR_YCrCb2BGR)

    denoised_yuv = np.stack([(y_denoised * 255).astype(np.uint8),
                             (cr * 255).astype(np.uint8),
                             (cb * 255).astype(np.uint8)], axis=2)
    denoised_bgr = cv2.cvtColor(denoised_yuv, cv2.COLOR_YCrCb2BGR)

    # 计算残差 (降噪去掉的部分)
    residual = np.abs(y_denoised - y)
    residual_vis = np.clip(residual * 5, 0, 1)  # 放大5倍

    # 转换各通道用于拼接
    y_orig_v = to_bgr_uint8(y)
    y_noisy_v = to_bgr_uint8(y_noise)  # 修复这里：之前误写成了 y
    y_denoised_v = to_bgr_uint8(y_denoised)
    residual_v = to_bgr_uint8(residual_vis)

    noise = y_noise - y # 直接计算噪声（不放大）
    noise_v = to_bgr_uint8(noise)

    # 拼接：第一行彩色，第二行亮度/残差
    row1 = np.hstack([img_bgr, noisy_bgr, denoised_bgr, noise_v])
    row2 = np.hstack([y_orig_v, y_noisy_v, y_denoised_v, residual_v])
    final_canvas = np.vstack([row1, row2])

    # 写标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        "Orig Color", "Noisy Color", "Denoised Color", "noisy",
        "Orig Y", "Noisy Y", "Denoised Y", "Residual (x10)"
    ]
    for i, text in enumerate(labels):
        tx = (i % 4) * w + 10
        ty = (i // 4) * h + 30
        cv2.putText(final_canvas, text, (tx, ty), font, 0.7, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(img_save_dir, f"full_process_{idx:02d}.png"), final_canvas)


# image_path = f"data/PhotoCD_PCD0992/03.png"
# y, cb, cr, img_bgr = read_png_to_yuv(image_path)
# sigma = 25
# sigma_norm = sigma / 255.0
# y_noise = add_poisson_gaussian_noise(y, a=0.02, sigma_norm=sigma_norm, seed=42)
# # BM3D 这里的 z 接受 [0,1] 范围
# denoised_y_norm = bm3d.bm3d(
#     z=y_noise,
#     sigma_psd=sigma_norm,
#     stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
# )
# y_denoised = np.clip(denoised_y_norm, 0, 1)
# # 调用显示函数
# # showPic(img_bgr, y, y_noise, cb, cr, y_denoised, img_save_dir, i)
# # 使用裁剪后的图像计算指标
# plt.imshow(y_denoised, cmap='gray');
# plt.axis('off')
# plt.tight_layout()
# plt.show()
# current_psnr = psnr(y, y_denoised, data_range=1.0)
# current_ssim = ssim(y, y_denoised, data_range=1.0)
# print(f"PSNR: {current_psnr:.2f} dB | SSIM: {current_ssim:.4f}\n")


dataset_path = "data/PhotoCD_PCD0992"
img_save_dir = Path(dataset_path) / "results"
img_save_dir.mkdir(parents=True, exist_ok=True)

# 换一个 CSV 文件名，以免覆盖你之前的并行版本结果
csv_file_path = Path(dataset_path) / "bm3d_lib_results.csv"

# 算法参数
sigma = 25
sigma_norm = sigma / 255.0
a_val = 0.02

# 打开 CSV 文件准备写入
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    # 写入表头，包含时间、PSNR、SSIM
    writer.writerow(['Image_Index', 'Time(s)', 'PSNR(dB)', 'SSIM'])

    # 循环遍历 1 到 23
    for idx in range(1, 24):
        print(f"\n{'=' * 20} 开始处理图片 {idx:02d} (官方库) {'=' * 20}")

        image_path = Path(dataset_path) / f"{idx:02d}.png"
        y, cb, cr, img_bgr = read_png_to_yuv(image_path)

        if img_bgr is None:
            print(f"警告：找不到路径为 {image_path} 的图片，跳过此图。")
            continue

        # 加噪
        y_noise = add_poisson_gaussian_noise(y, a=a_val, sigma_norm=sigma_norm, seed=42)

        # 记录开始时间
        t_start = time.time()

        # BM3D 这里的 z 接受 [0,1] 范围
        denoised_y_norm = bm3d.bm3d(
            z=y_noise,
            sigma_psd=sigma_norm,
            stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
        )
        y_denoised = np.clip(denoised_y_norm, 0, 1)

        # 记录耗时
        duration = time.time() - t_start
        print(f"库函数处理完成，耗时: {duration:.2f}s")

        # 计算指标
        current_psnr = psnr(y, y_denoised, data_range=1.0)
        current_ssim = ssim(y, y_denoised, data_range=1.0)
        print(f"PSNR: {current_psnr:.2f} dB | SSIM: {current_ssim:.4f}")

        # 写入 CSV 并实时保存
        writer.writerow([idx, round(duration, 2), round(current_psnr, 2), round(current_ssim, 4)])
        csv_file.flush()

        # 可选：如果还需要调用你的详细比对保存函数，取消下面这行注释
        # showPic(img_bgr, y, y_noise, cb, cr, y_denoised, img_save_dir, idx)

        # 绘制当前降噪图并保存，替代阻塞程序的 plt.show()
        plt.figure(figsize=(6, 6))
        plt.imshow(y_denoised, cmap='gray')
        plt.axis('off')
        plt.tight_layout()

        plot_save_path = img_save_dir / f"{idx:02d}_lib_denoised.png"
        plt.savefig(plot_save_path)
        plt.close()  # 防止内存泄漏

print(f"\n所有处理已完成，库函数结果已保存至：{csv_file_path}")