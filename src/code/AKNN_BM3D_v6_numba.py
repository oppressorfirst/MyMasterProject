import os
import cv2
import csv
import time
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Numba 核心加速库
from numba import njit, prange

# 引入 3D 变换库
from scipy.fft import dctn, idctn
import pywt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ==========================================
# 工具函数 (VST, 分块等保持不变)
# ==========================================

def forward_gat(z, a, sigma):
    return 2.0 * np.sqrt(np.maximum(z / a + 3.0 / 8.0 + (sigma ** 2) / (a ** 2), 0))


def inverse_gat(D, a, sigma):
    return a * ((D / 2.0) ** 2 - 1.0 / 8.0 - (sigma ** 2) / (a ** 2))


def split_image_into_4_blocks(img, overlap=39):
    H, W = img.shape[:2]
    mid_H, mid_W = H // 2, W // 2
    coords = [
        (0, mid_H + overlap, 0, mid_W + overlap),
        (0, mid_H + overlap, mid_W - overlap, W),
        (mid_H - overlap, H, 0, mid_W + overlap),
        (mid_H - overlap, H, mid_W - overlap, W)
    ]
    blocks = [img[y0:y1, x0:x1].copy() for (y0, y1, x0, x1) in coords]
    return blocks, coords


def read_png_to_yuv(path, normalize=True):
    img = cv2.imread(path)
    if img is None: return None, None, None, None
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = yuv[:, :, 0], yuv[:, :, 1], yuv[:, :, 2]
    if normalize:
        y = y.astype(np.float32) / 255.0
        cb = cb.astype(np.float32) / 255.0
        cr = cr.astype(np.float32) / 255.0
    return y, cb, cr, img


def add_poisson_gaussian_noise(img_clean, a=0.1, sigma_norm=25 / 255, seed=None):
    rng = np.random.default_rng(seed)
    photon_counts = np.maximum(img_clean / a, 1e-10)
    noisy_poisson = rng.poisson(photon_counts) * a
    noisy_gaussian = rng.normal(0, sigma_norm, img_clean.shape)
    return np.clip(noisy_poisson + noisy_gaussian, 0.0, 1.0).astype(np.float32)


# ==========================================
# 【核心重写】Numba 加速的 AKNN 模块
# ==========================================

@njit(fastmath=True)
def compute_patch_distance_numba(img, y, x, ny, nx, patch_size):
    """底层 SSD 计算，剔除 Python 上下文，纯 C 级别执行"""
    r = patch_size // 2
    dist = 0.0
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            diff = img[y + i, x + j] - img[ny + i, nx + j]
            dist += diff * diff
    return dist


@njit(fastmath=True)
def update_best_k_numba(img, y, x, prop_dy, prop_dx, offsets, dists, patch_size, K):
    """用插入排序维护优先队列，消除 Python 列表排序开销"""
    h, w = img.shape
    r = patch_size // 2
    ny, nx = y + prop_dy, x + prop_dx

    # 严格的边界检查
    if y - r < 0 or y + r >= h or x - r < 0 or x + r >= w: return
    if ny - r < 0 or ny + r >= h or nx - r < 0 or nx + r >= w: return

    # 排除自己及其周围极其相近的区域
    if (prop_dy != 0 or prop_dx != 0) and (abs(prop_dy) <= r and abs(prop_dx) <= r): return

    # 查重检查 (避免同一队列里存在重复项)
    for k in range(K):
        if offsets[y, x, k, 0] == prop_dy and offsets[y, x, k, 1] == prop_dx:
            return

    # 计算距离
    new_dist = compute_patch_distance_numba(img, y, x, ny, nx, patch_size)

    # 早期退出：比队列里最差的还差，直接扔掉
    if new_dist >= dists[y, x, K - 1]:
        return

    # 插入排序
    insert_pos = -1
    for k in range(K):
        if new_dist < dists[y, x, k]:
            insert_pos = k
            break

    if insert_pos != -1:
        # 数据后移
        for k in range(K - 1, insert_pos, -1):
            dists[y, x, k] = dists[y, x, k - 1]
            offsets[y, x, k, 0] = offsets[y, x, k - 1, 0]
            offsets[y, x, k, 1] = offsets[y, x, k - 1, 1]

        # 填入新数据
        dists[y, x, insert_pos] = new_dist
        offsets[y, x, insert_pos, 0] = prop_dy
        offsets[y, x, insert_pos, 1] = prop_dx


@njit(parallel=True, fastmath=True)
def initialize_aknn_numba(img, K, patch_size, step):
    """并行化初始化，支持多核"""
    H, W = img.shape
    r = patch_size // 2
    sigma_s = W / 3.0

    nn_offsets = np.zeros((H, W, K, 2), dtype=np.int32)
    nn_dists = np.full((H, W, K), np.inf, dtype=np.float32)

    for y in prange(r, H - r):
        if (y - r) % step != 0: continue
        for x in range(r, W - r):
            if (x - r) % step != 0: continue

            # 多找几个随机点确保填满 K
            for k in range(K * 2):
                dy = int(np.round(sigma_s * np.random.randn()))
                dx = int(np.round(sigma_s * np.random.randn()))
                ny, nx = y + dy, x + dx

                # 若越界则修正为内部随机点
                if ny - r < 0 or ny + r >= H or nx - r < 0 or nx + r >= W:
                    ny = np.random.randint(r, H - r)
                    nx = np.random.randint(r, W - r)
                    dy, dx = ny - y, nx - x

                update_best_k_numba(img, y, x, dy, dx, nn_offsets, nn_dists, patch_size, K)

    return nn_offsets, nn_dists


@njit(fastmath=True)
def propagation_step_numba(img, offsets, dists, patch_size, iter_num, step):
    """时空相依无法无脑并行，采用快速编译执行"""
    H, W = img.shape
    K = offsets.shape[2]
    r = patch_size // 2

    if iter_num % 2 == 0:
        # 正向：看左上
        for y in range(r + step, H - r, step):
            for x in range(r + step, W - r, step):
                # 检查左边邻居
                for k in range(K):
                    dy, dx = offsets[y, x - step, k]
                    update_best_k_numba(img, y, x, dy, dx, offsets, dists, patch_size, K)
                # 检查上边邻居
                for k in range(K):
                    dy, dx = offsets[y - step, x, k]
                    update_best_k_numba(img, y, x, dy, dx, offsets, dists, patch_size, K)
    else:
        # 反向：看右下
        end_y = r + ((H - r - 1 - r) // step) * step
        end_x = r + ((W - r - 1 - r) // step) * step
        for y in range(end_y - step, r - 1, -step):
            for x in range(end_x - step, r - 1, -step):
                # 检查右边邻居
                for k in range(K):
                    dy, dx = offsets[y, x + step, k]
                    update_best_k_numba(img, y, x, dy, dx, offsets, dists, patch_size, K)
                # 检查下边邻居
                for k in range(K):
                    dy, dx = offsets[y + step, x, k]
                    update_best_k_numba(img, y, x, dy, dx, offsets, dists, patch_size, K)


@njit(parallel=True, fastmath=True)
def random_search_step_numba(img, offsets, dists, patch_size, search_radius, step):
    """随机搜索支持多核并行处理"""
    H, W = img.shape
    K = offsets.shape[2]
    r = patch_size // 2

    for y in prange(r, H - r):
        if (y - r) % step != 0: continue
        for x in range(r, W - r):
            if (x - r) % step != 0: continue

            for k in range(K):
                best_dy, best_dx = offsets[y, x, k, 0], offsets[y, x, k, 1]
                rand_dy = int(np.round(search_radius * np.random.randn()))
                rand_dx = int(np.round(search_radius * np.random.randn()))

                search_dy, search_dx = best_dy + rand_dy, best_dx + rand_dx
                update_best_k_numba(img, y, x, search_dy, search_dx, offsets, dists, patch_size, K)


def run_aknn_fast(img, iterations, K, patch_size, step):
    """供外部调用的 Python 封装器"""
    t_start = time.time()

    # 将输入图片保证为 numba 友好的连续 float32 数组
    img_fast = np.ascontiguousarray(img, dtype=np.float32)

    offsets, dists = initialize_aknn_numba(img_fast, K, patch_size, step)
    search_radius = float(img.shape[1])

    for i in range(iterations):
        propagation_step_numba(img_fast, offsets, dists, patch_size, i, step)

        current_radius = max(search_radius * (0.5 ** i), 1.0)
        random_search_step_numba(img_fast, offsets, dists, patch_size, current_radius, step)

    # print(f"  AKNN completed in {time.time() - t_start:.2f}s")
    return offsets, dists


# ==========================================
# 降噪管道和可视化
# ==========================================

def bm3d_1st_stage_vst_offsets(img_vst, offsets, patch_size, step=3):
    H, W = img_vst.shape
    K_offsets = offsets.shape[2]
    r = patch_size // 2

    numerator = np.zeros_like(img_vst, dtype=np.float64)
    denominator = np.zeros_like(img_vst, dtype=np.float64)

    sigma_vst = 1.0
    lambda_3d = 2.7 * sigma_vst
    sigma_vst2 = sigma_vst ** 2

    # Scipy 库不支持 numba，所以这部分用普通的 Python 循环执行
    # 由于步长(step)的存在，这部分的耗时其实非常短
    for y in range(r, H - r, step):
        for x in range(r, W - r, step):
            coords = [(y, x)]

            for k in range(K_offsets):
                dy, dx = offsets[y, x, k]
                ny, nx = y + dy, x + dx
                if r <= ny <= H - patch_size + r and r <= nx <= W - patch_size + r:
                    coords.append((ny, nx))

            K_actual = len(coords)
            if K_actual <= 1: continue

            group_3d = np.zeros((K_actual, patch_size, patch_size), dtype=np.float64)
            for i, (cy, cx) in enumerate(coords):
                group_3d[i] = img_vst[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size]

            # 3D 变换
            group_2d_dct = dctn(group_3d, axes=(1, 2), norm='ortho')
            haar_coeffs = pywt.wavedec(group_2d_dct, 'haar', mode='symmetric', axis=0)

            # 阈值截断
            n_nonzero = 0
            for i in range(len(haar_coeffs)):
                haar_coeffs[i][np.abs(haar_coeffs[i]) < lambda_3d] = 0
                n_nonzero += np.sum(haar_coeffs[i] != 0)

            weight = 1.0 / (n_nonzero * sigma_vst2) if n_nonzero > 0 else 1.0 / sigma_vst2

            # 逆变换
            group_1d_inv = pywt.waverec(haar_coeffs, 'haar', mode='symmetric', axis=0)
            group_1d_inv = group_1d_inv[:K_actual, :, :]
            group_3d_denoised = idctn(group_1d_inv, axes=(1, 2), norm='ortho')

            # 聚合贴回原图
            for i, (cy, cx) in enumerate(coords):
                numerator[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size] += group_3d_denoised[i] * weight
                denominator[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size] += weight

    mask = denominator > 0
    denoised_img = img_vst.copy()
    denoised_img[mask] = numerator[mask] / denominator[mask]

    return denoised_img


def process_single_block(block_idx, noisy_vst_block, guide_vst_block, K, patch_size, process_step):
    """集成调用"""
    print(f"\n--- [Worker {block_idx}] 开始处理 ---")

    # 替换为高速 Numba 版本的 AKNN
    final_offsets, final_dists = run_aknn_fast(
        guide_vst_block, iterations=2, K=K, patch_size=patch_size, step=process_step
    )

    denoised_vst_block = bm3d_1st_stage_vst_offsets(
        img_vst=noisy_vst_block,
        offsets=final_offsets,
        patch_size=patch_size,
        step=process_step
    )

    print(f"--- [Worker {block_idx}] 处理完成 ---")
    return block_idx, denoised_vst_block


