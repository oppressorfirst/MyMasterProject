import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import njit, prange
from src.utils import getMetrics, AI_Metrics
from tqdm import tqdm
from pathlib import Path

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


def non_local_means(noisy_img, h, patch_r, window_r, sigma):
    H, W = noisy_img.shape
    output_img = np.zeros([H, W])

    padded_img = np.pad(noisy_img, pad_width=patch_r, mode='reflect')

    kernel_a = make_kernel(patch_r, sigma)
    print("开始处理... (可能需要一点时间)")
    pbar = tqdm(total=H * W, desc="NLM Pixels")
    for i_row in range(H):
        for i_col in range(W):
            pbar.update(1)
            # 这里的 i 对应文本中的像素 i
            # 在 padded_img 中，i 的坐标需要偏移 f
            # print(f"match for  {i_row}, {i_col}")
            i_row_pad = i_row + patch_r
            i_col_pad = i_col + patch_r

            patch_i = padded_img[i_row_pad - patch_r: i_row_pad + patch_r + 1,
            i_col_pad - patch_r: i_col_pad + patch_r + 1]

            w_sum = 0.0  # 对应文本中的 Z(i)
            weighted_val = 0.0  # 对应 ∑ w(i,j)v(j)

            r_min_row = max(i_row - window_r, 0)
            r_max_row = min(i_row + window_r, H)
            r_min_col = max(i_col - window_r, 0)
            r_max_col = min(i_col + window_r, W)

            for j_row in range(r_min_row , r_max_row):
                for j_col in range(r_min_col , r_max_col):

                    j_row_pad = j_row + patch_r
                    j_col_pad = j_col + patch_r
                    patch_j = padded_img[j_row_pad - patch_r: j_row_pad + patch_r + 1,
                                            j_col_pad - patch_r: j_col_pad + patch_r + 1]
                    distance_squared = np.sum(((patch_i - patch_j) ** 2) * kernel_a)
                    weight = np.exp(-distance_squared / (h ** 2))

                    weighted_val += weight * noisy_img[j_row, j_col]  # 注意这里乘的是 v(j)
                    w_sum += weight

            output_img[i_row, i_col] = weighted_val / w_sum
    pbar.close()
    return output_img





original_path = DATA_DIR / "classic_photo" / "lena_gray.png"
noisy_path = DATA_DIR / "classic_photo_AWGN_sigma20_seed123456" / "lena_gray_sigma20_seed123456.png"

original_img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)


patch_r = 3
window_r = 20
h_val = 15
sigma = 1.1


output_filename = OUT_DIR/ "images"/ "baseline"/"NLM"/f"lena_gray_NLM_h{h_val}_patch_r{patch_r}_window_r{window_r}_sigma{sigma}.png"
output_filename.parent.mkdir(parents=True, exist_ok=True)

# 确保图片读取成功
if original_img is None or noisy_img is None:
    print("错误：找不到图片，请检查路径。")
else:
    original_img = original_img.astype(np.float32)
    noisy_img = noisy_img.astype(np.float32)
    # output_img = cv2.imread(str(output_filename), cv2.IMREAD_GRAYSCALE)

    output_img = non_local_means(
        noisy_img.astype(np.float32),
        h = h_val,
        patch_r = patch_r,
       window_r= window_r,
        sigma=sigma
    )
    output_img_uint8 = np.clip(output_img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_filename), output_img_uint8)

    # 6. 计算指标 (Noise vs Original)
    print("-" * 30)
    noise_metrics = getMetrics.calculate_metrics(original_img.astype(np.uint8), noisy_img.astype(np.uint8))
    print(f"【原带噪图】 PSNR: {noise_metrics['PSNR']:.2f} | SSIM: {noise_metrics['SSIM']:.4f}")

    # 7. 计算指标 (Denoised vs Original)
    denoised_metrics = getMetrics.calculate_metrics(original_img.astype(np.uint8), output_img)
    print(f"【去噪声后】 PSNR: {denoised_metrics['PSNR']:.2f} | SSIM: {denoised_metrics['SSIM']:.4f}")
    print("-" * 30)
    print(f"处理完成！图片已保存为: {output_filename}")

    lpips, _ = AI_Metrics.compare_advanced_metrics(str(original_path), str(output_filename))
    print(f"{lpips:.4f} ")

