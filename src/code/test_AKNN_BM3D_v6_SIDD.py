
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
import pywt

from src.code.AKNN_BM3D_v6 import read_png_to_yuv, add_poisson_gaussian_noise, split_image_into_4_blocks, \
    process_single_block, inverse_gat, forward_gat, showPic


dataset_path = "data/SIDD_Medium_Srgb"
scene_id ="0013_001_S6_03200_01250_3200_L"
clean_path = Path(dataset_path) / Path(scene_id) /  "0013_GT_SRGB_010.PNG"
noisy_path = Path(dataset_path) / Path(scene_id) /  "0013_NOISY_SRGB_010.PNG"
img_save_dir =Path(dataset_path) / Path(scene_id) / "results"

y, cb, cr, clean_img_cv = read_png_to_yuv(clean_path)
y_noisy, cb_noisy, cr_noisy, noisy_img_cv = read_png_to_yuv(noisy_path)

if clean_img_cv is None:
    print(f"错误：找不到路径为 {clean_path} 的图片，请检查路径。")
    exit()
if noisy_img_cv is None:
    print(f"错误：找不到路径为 {clean_path} 的图片，请检查路径。")
    exit()

sigma_val = 25
sigma_norm = sigma_val / 255.0
a_val = 0.02
K = 7
patch_size = 7
process_step = 2
overlap_pixels = 39  # 设置你想要的重叠像素

np.random.seed(42)
y_noisy = add_poisson_gaussian_noise(y, a=a_val, sigma_norm=sigma_norm, seed=42)
guide_img = cv2.GaussianBlur(y_noisy, (5, 5), 1.5)

# ==========================================
# 分治策略 (Divide and Conquer)
# ==========================================

# 1. 切分为 4 块
noisy_blocks, block_coords = split_image_into_4_blocks(y_noisy, overlap=overlap_pixels)
guide_blocks, _ = split_image_into_4_blocks(guide_img, overlap=overlap_pixels)

denoised_blocks = [None] * 4

# 2. 并行处理 (使用 4 个进程)
print("启动 4 进程并行处理...")
t_start_parallel = time.time()

# ProcessPoolExecutor 可以绕过 Python 的 GIL，实现真正的多核计算
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    # 提交 4 个任务
    futures = []
    for i in range(4):
        future = executor.submit(
            process_single_block,
            i, noisy_blocks[i], guide_blocks[i],
            K, patch_size, process_step, sigma_norm, a_val
        )
        futures.append(future)

    # 收集结果
    for future in concurrent.futures.as_completed(futures):
        block_idx, result_block = future.result()
        denoised_blocks[block_idx] = result_block

print(f"并行处理完成，耗时: {time.time() - t_start_parallel:.2f}s")

# 3. 图像融合 (Merge)
H, W = y.shape
numerator = np.zeros((H, W), dtype=np.float32)
denominator = np.zeros((H, W), dtype=np.float32)

for i in range(4):
    y0, y1, x0, x1 = block_coords[i]

    # 直接把处理好的子图加进去，不需要任何 mask！
    numerator[y0:y1, x0:x1] += denoised_blocks[i]

    # 这个区域的计数器统一加 1
    denominator[y0:y1, x0:x1] += 1.0

    # 取平均：
    # 重叠区域由于被加了多次，denominator 自然会是 2 或 4
    # 边缘和非重叠区域 denominator 自然是 1
y_denoised = numerator / denominator
y_denoised = np.clip(y_denoised, 0, 1)

# ==========================================
# 评估与可视化
# ==========================================
showPic(clean_img_cv, y, y_noisy, cb, cr, y_denoised, img_save_dir, idx)
current_psnr = psnr(y, y_denoised, data_range=1.0)
current_ssim = ssim(y, y_denoised, data_range=1.0)
print(f"\nFinal PSNR: {current_psnr:.2f} dB | SSIM: {current_ssim:.4f}\n")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Clean (Ground Truth)")
plt.imshow(y, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"Noisy (Sigma={sigma_val})")
plt.imshow(y_noisy, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Parallel BM3D 1st Stage")
plt.imshow(y_denoised, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()