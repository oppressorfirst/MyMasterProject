import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import time
import concurrent.futures
import itertools  # 【新增】用于生成笛卡尔积（参数组合）
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import sys
print(sys.path)

# 这里假设您已经导入了自定义的函数

from src.code.AKNN_BM3D_v6_numba import read_png_to_yuv, add_poisson_gaussian_noise, split_image_into_4_blocks, process_single_block, inverse_gat, forward_gat
dataset_path =  "data/PhotoCD_PCD0992"
img_save_dir = Path(dataset_path) / "results_grid_search"
img_save_dir.mkdir(parents=True, exist_ok=True)

# CSV 保存路径
csv_file_path = Path(dataset_path) / "AKNN_grid_search_results.csv"

# ==========================================
# 【改动 1】定义需要搜索的参数列表
# ==========================================
K_list = [7, 15]
a_val_list = [0.005, 0.01, 0.02, 0.03]  # a_val 从 0.005 到 0.03 的离散取值
patch_size_list = [5, 7, 9, 11]
process_step_list = [1, 2, 3, 4, 5, 6]
overlap_pixels_list = [39, 59]
sigma_val_list = [15, 25, 50]

# 生成所有参数组合
param_combinations = list(itertools.product(
    sigma_val_list, a_val_list, K_list, patch_size_list, process_step_list, overlap_pixels_list
))
total_combinations = len(param_combinations)

# 【强烈建议】为了防止网格搜索生成几万张图片撑爆硬盘，默认关闭画图。只记录指标到 CSV。
SAVE_IMAGES = False
# 【强烈建议】测试时可以先将范围缩小，例如 range(1, 4) 只测前三张图
TEST_IMAGE_RANGE = range(1, 24)

print(f"总计需要测试的参数组合数: {total_combinations} 种。")
print("正在将测试图片预加载至内存...")
image_cache = {}
for idx in TEST_IMAGE_RANGE:
    clean_path = Path(dataset_path) / f"{idx:02d}.png"
    y, cb, cr, clean_img_cv = read_png_to_yuv(clean_path)
    if clean_img_cv is not None:
        image_cache[idx] = (y, cb, cr, clean_img_cv)
print(f"预加载完成，共计 {len(image_cache)} 张图片。")

# ==========================================
# 极致优化 2：在最外层创建一个长存活期的进程池
# ==========================================
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file, \
        concurrent.futures.ProcessPoolExecutor(max_workers=4) as global_executor:
    writer = csv.writer(csv_file)
    writer.writerow([
        'Sigma', 'a_val', 'K', 'Patch_Size', 'Step', 'Overlap',
        'Image_Index', 'Time(s)', 'PSNR(dB)', 'SSIM'
    ])

    combination_count = 0

    for params in param_combinations:
        sigma_val, a_val, K, patch_size, process_step, overlap_pixels = params
        sigma_norm = sigma_val / 255.0
        combination_count += 1

        print(
            f"\n[{combination_count}/{total_combinations}] 测试参数: Sigma={sigma_val}, a={a_val}, K={K}, patch={patch_size}, step={process_step}, overlap={overlap_pixels}")

        for idx, (y, cb, cr, clean_img_cv) in image_cache.items():

            # 【重要】每次都必须重新设置随机种子，保证针对同一张图不同参数时，加入的噪声完全一样，控制变量法才有效
            np.random.seed(42)
            y_noisy = add_poisson_gaussian_noise(y, a=a_val, sigma_norm=sigma_norm, seed=42)

            y_noisy_vst = forward_gat(y_noisy, a=a_val, sigma=sigma_norm)
            guide_img_vst = cv2.GaussianBlur(y_noisy_vst, (5, 5), 1.5)

            noisy_blocks, block_coords = split_image_into_4_blocks(y_noisy_vst, overlap=overlap_pixels)
            guide_blocks, _ = split_image_into_4_blocks(guide_img_vst, overlap=overlap_pixels)
            denoised_vst_blocks = [None] * 4

            t_start_parallel = time.time()

            # 派发任务给外层的全局 global_executor，不再频繁销毁重建！
            futures = []
            for i in range(4):
                future = global_executor.submit(
                    process_single_block,
                    i, noisy_blocks[i], guide_blocks[i],
                    K, patch_size, process_step
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                block_idx, result_block = future.result()
                denoised_vst_blocks[block_idx] = result_block

            duration = time.time() - t_start_parallel

            # 3. 图像融合...
            H, W = y.shape
            numerator = np.zeros((H, W), dtype=np.float32)
            denominator = np.zeros((H, W), dtype=np.float32)
            for i in range(4):
                y0, y1, x0, x1 = block_coords[i]
                numerator[y0:y1, x0:x1] += denoised_vst_blocks[i]
                denominator[y0:y1, x0:x1] += 1.0

            y_denoised_vst = numerator / denominator
            y_denoised = inverse_gat(y_denoised_vst, a=a_val, sigma=sigma_norm)
            y_denoised = np.clip(y_denoised, 0.0, 1.0)

            # 评估...
            current_psnr = psnr(y, y_denoised, data_range=1.0)
            current_ssim = ssim(y, y_denoised, data_range=1.0)

            print(
                f"  --> Img {idx:02d} | 耗时: {duration:.2f}s | PSNR: {current_psnr:.2f} dB | SSIM: {current_ssim:.4f}")

            writer.writerow([
                sigma_val, a_val, K, patch_size, process_step, overlap_pixels,
                idx, round(duration, 2), round(current_psnr, 2), round(current_ssim, 4)
            ])
            csv_file.flush()