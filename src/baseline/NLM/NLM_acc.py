import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import njit, prange
from src.utils import getMetrics, AI_Metrics
from tqdm import tqdm
from pathlib import Path
from numba import njit, prange
import time

# 定位到项目根 MyMasterProject
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "out"

def make_kernel(r, sigma):
    """
    创建距离加权核
    r: patch 半径 (例如 patch 大小为 (2r+1)x(2r+1))
    sigma: 高斯标准差
    """
    if sigma == 0:
        return np.ones((2 * r + 1, 2 * r + 1)) / ((2 * r + 1) ** 2)

    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)  # 归一化


@njit(parallel=True, fastmath=True)
def non_local_means_numba(noisy_img, padded, kernel, h, patch_r, window_r):
    H, W = noisy_img.shape
    output_img = np.zeros((H, W), dtype=np.float32)

    for i_row in prange(H):
        for i_col in range(W):
            i_row_pad = i_row + patch_r
            i_col_pad = i_col + patch_r

            w_sum = 0.0
            weighted_val = 0.0

            r_min_row = max(i_row - window_r, 0)
            r_max_row = min(i_row + window_r, H)
            r_min_col = max(i_col - window_r, 0)
            r_max_col = min(i_col + window_r, W)

            for j_row in range(r_min_row, r_max_row):
                for j_col in range(r_min_col, r_max_col):

                    j_row_pad = j_row + patch_r
                    j_col_pad = j_col + patch_r

                    dist2 = 0.0

                    for dx in range(-patch_r, patch_r + 1):
                        for dy in range(-patch_r, patch_r + 1):
                            a = padded[i_row_pad + dx, i_col_pad + dy]
                            b = padded[j_row_pad + dx, j_col_pad + dy]
                            w = kernel[dx + patch_r, dy + patch_r]
                            diff = a - b
                            dist2 += w * diff * diff

                    weight = np.exp(-dist2 / (h * h))
                    weighted_val += weight * noisy_img[j_row, j_col]
                    w_sum += weight

            if w_sum > 1e-8:
                output_img[i_row, i_col] = weighted_val / w_sum
            else:
                output_img[i_row, i_col] = noisy_img[i_row, i_col]

    return output_img




original_path = DATA_DIR / "classic_photo" / "lena_gray.png"
noisy_path = DATA_DIR / "classic_photo_AWGN_sigma20_seed123456" / "lena_gray_sigma20_seed123456.png"

original_img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)

h_val=20
patch_r=4
window_r=20
sigma=1

output_filename = OUT_DIR/ "images"/"baseline"/"NLM"/f"lena_gray_NLM_acc_h{h_val}_patch_r{patch_r}_window_r{window_r}_sigma{sigma}.png"
output_filename.parent.mkdir(parents=True, exist_ok=True)

# 确保图片读取成功
if original_img is None or noisy_img is None:
    print("错误：找不到图片，请检查路径。")
else:
    original_img = original_img.astype(np.float32)
    noisy_img = noisy_img.astype(np.float32)

    # output_img = non_local_means(
    #     noisy_img.astype(np.float32),
    #     h = h_val,
    #     patch_r = patch_r,
    #    window_r= window_r,
    #     sigma=sigma
    # )

    padded_img = np.pad(noisy_img, pad_width=patch_r, mode='reflect')
    # --- 开始计时 ---
    start_time = time.perf_counter()
    output_img = non_local_means_numba(
        noisy_img.astype(np.float32),
        padded_img,
        make_kernel(patch_r, sigma),
        h_val,
        patch_r,
        window_r
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"NLM 处理耗时: {elapsed_time:.4f} 秒")
    output_img_uint8 = np.clip(output_img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_filename), output_img_uint8)

    # 6. 计算指标 (Noise vs Original)
    print("-" * 30)
    noise_metrics = getMetrics.calculate_metrics(original_img.astype(np.uint8), noisy_img.astype(np.uint8))
    print(f"【原带噪图】 PSNR: {noise_metrics['PSNR']:.2f} | SSIM: {noise_metrics['SSIM']:.4f}")

    # 7. 计算指标 (Denoised vs Original)
    denoised_metrics = getMetrics.calculate_metrics(original_img.astype(np.uint8), output_img)
    print(f"【双边滤波】 PSNR: {denoised_metrics['PSNR']:.2f} | SSIM: {denoised_metrics['SSIM']:.4f}")
    print("-" * 30)
    print(f"处理完成！图片已保存为: {output_filename}")

    lpips, _ = AI_Metrics.compare_advanced_metrics(str(original_path), str(output_filename))
    print(f"{lpips:.4f} ")

