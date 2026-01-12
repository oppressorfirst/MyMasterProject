import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import njit, prange

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


# def non_local_means(noisy_img, h, patch_r, window_r, sigma):
#     H, W = noisy_img.shape
#     output_img = np.zeros([H, W])
#
#     padded_img = np.pad(noisy_img, pad_width=patch_r, mode='reflect')
#
#     kernel_a = make_kernel(patch_r, sigma)
#     print("开始处理... (可能需要一点时间)")
#     for i_row in range(H):
#         for i_col in range(W):
#             # 这里的 i 对应文本中的像素 i
#             # 在 padded_img 中，i 的坐标需要偏移 f
#             print(f"match for  {i_row}, {i_col}")
#             i_row_pad = i_row + patch_r
#             i_col_pad = i_col + patch_r
#
#             patch_i = padded_img[i_row_pad - patch_r: i_row_pad + patch_r + 1,
#             i_col_pad - patch_r: i_col_pad + patch_r + 1]
#
#             w_sum = 0.0  # 对应文本中的 Z(i)
#             weighted_val = 0.0  # 对应 ∑ w(i,j)v(j)
#
#             for j_row in range(H):
#                 for j_col in range(W):
#
#                     j_row_pad = j_row + patch_r
#                     j_col_pad = j_col + patch_r
#                     patch_j = padded_img[j_row_pad - patch_r: j_row_pad + patch_r + 1,
#                                             j_col_pad - patch_r: j_col_pad + patch_r + 1]
#                     distance_squared = np.sum(((patch_i - patch_j) ** 2) * kernel_a)
#                     weight = np.exp(-distance_squared / (h ** 2))
#
#                     weighted_val += weight * noisy_img[j_row, j_col]  # 注意这里乘的是 v(j)
#                     w_sum += weight
#
#             output_img[i_row, i_col] = weighted_val / w_sum
#     return output_img

from numba import njit, prange
import numpy as np

@njit(parallel=True, fastmath=True)
def non_local_means_numba(noisy_img, padded, kernel_a, h, patch_r):
    H, W = noisy_img.shape
    output_img = np.zeros((H, W), dtype=np.float32)

    for i_row in prange(H):
        for i_col in range(W):
            print(f"match for  {i_row}, {i_col}")
            i_row_pad = i_row + patch_r
            i_col_pad = i_col + patch_r

            w_sum = 0.0
            weighted_val = 0.0

            for j_row in range(H):
                for j_col in range(W):

                    j_row_pad = j_row + patch_r
                    j_col_pad = j_col + patch_r

                    dist2 = 0.0

                    for dx in range(-patch_r, patch_r + 1):
                        for dy in range(-patch_r, patch_r + 1):
                            a = padded[i_row_pad + dx, i_col_pad + dy]
                            b = padded[j_row_pad + dx, j_col_pad + dy]
                            w = kernel_a[dx + patch_r, dy + patch_r]
                            diff = a - b
                            dist2 += w * diff * diff

                    weight = np.exp(-dist2 / (h * h))
                    weighted_val += weight * noisy_img[j_row, j_col]
                    w_sum += weight

            output_img[i_row, i_col] = weighted_val / w_sum

    return output_img




original_path = "../../../data/classic_photo/lena_gray.png"
noisy_path = "../../../data/classic_photo_AWGN_sigma20_seed123456/lena_gray_sigma20_seed123456.png"

original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)

# 确保图片读取成功
if original_img is None or noisy_img is None:
    print("错误：找不到图片，请检查路径。")
else:
    original_img = original_img.astype(np.float32)
    noisy_img = noisy_img.astype(np.float32)
    patch_r = 2
    kernel_a = make_kernel(r=patch_r, sigma=1).astype(np.float32)

    padded = np.pad(noisy_img, patch_r, mode='reflect').astype(np.float32)

    output_img = non_local_means_numba(
        noisy_img.astype(np.float32),
        padded,
        kernel_a,
        h=0.1,
        patch_r=patch_r
    )
