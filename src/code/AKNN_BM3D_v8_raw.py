import os
import glob
import re
import h5py
import scipy.io as sio
import pandas as pd
import cv2
import numpy as np
import time
import csv
import pywt
import sys
import concurrent.futures
from pathlib import Path
from tqdm import tqdm, trange
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.fft import dctn, idctn

# =====================================================================
# 1. 基础变换工具 (VST)
# =====================================================================
def forward_gat(z, a, sigma):
    return 2.0 * np.sqrt(np.maximum(z / a + 3.0 / 8.0 + (sigma ** 2) / (a ** 2), 0))

def inverse_gat(D, a, sigma):
    return a * ((D / 2.0) ** 2 - 1.0 / 8.0 - (sigma ** 2) / (a ** 2))

# =====================================================================
# 2. Bayer 打包与解包
# =====================================================================
def pack_bayer(raw_img):
    """
    将 (H, W) RAW 图像按空间位置打包为 (H/2, W/2, 4)。
    通道顺序: 0=(偶行,偶列), 1=(偶行,奇列), 2=(奇行,偶列), 3=(奇行,奇列)
    """
    H, W = raw_img.shape
    packed = np.zeros((H // 2, W // 2, 4), dtype=raw_img.dtype)
    packed[:, :, 0] = raw_img[0::2, 0::2]
    packed[:, :, 1] = raw_img[0::2, 1::2]
    packed[:, :, 2] = raw_img[1::2, 0::2]
    packed[:, :, 3] = raw_img[1::2, 1::2]
    return packed

def unpack_bayer(packed):
    """将 (H/2, W/2, 4) 还原为 (H, W) RAW 图像"""
    H2, W2, C = packed.shape
    raw_img = np.zeros((H2 * 2, W2 * 2), dtype=packed.dtype)
    raw_img[0::2, 0::2] = packed[:, :, 0]
    raw_img[0::2, 1::2] = packed[:, :, 1]
    raw_img[1::2, 0::2] = packed[:, :, 2]
    raw_img[1::2, 1::2] = packed[:, :, 3]
    return raw_img

# =====================================================================
# 3. 噪声参数读取 (SIDD 格式)
# =====================================================================
def get_noise_and_bayer_info(scene_id, noise_csv_path, bayer_csv_path):
    """
    不仅返回噪声参数，还返回绿通道 (Gr) 在 4 个 Packed 通道中的索引。
    """
    camera_id = scene_id.split('_')[2]

    df_bayer = pd.read_csv(bayer_csv_path)
    bayer_pattern = df_bayer[df_bayer['camera_id'] == camera_id]['bayer_pattern'].values[0].lower()

    color_map = {
        'rggb': ['r', 'g', 'g', 'b'],
        'bggr': ['b', 'g', 'g', 'r'],
        'grbg': ['g', 'r', 'b', 'g'],
        'gbrg': ['g', 'b', 'r', 'g']
    }
    channel_colors = color_map[bayer_pattern]
    
    # 【新增】：找到第一个绿通道（Gr）的索引
    gr_index = channel_colors.index('g')

    df_noise = pd.read_csv(noise_csv_path)
    scene_row = df_noise[df_noise['scene_instance_id'] == scene_id].iloc[0]

    params = []
    for color in channel_colors:
        beta1 = scene_row[f'beta1_{color}']
        beta2 = scene_row[f'beta2_{color}']
        a_est = beta1
        sigma_est = np.sqrt(max(beta2, 0.0))
        params.append((a_est, sigma_est))

    return params, bayer_pattern, gr_index
# =====================================================================
# 4. 硬件架构映射：Tile-based 切分与重组 (Core + Halo)
# =====================================================================
def split_image_into_grid(img, block_size=166, overlap=38):
    """
    完美模拟硬件的 Core Tile + Halo Margin 架构。
    支持单通道 (H, W) 和多通道 (H, W, C) 输入。
    当 block_size=166, overlap=38 时，Stride=128，核心区为 128x128。
    """
    H, W = img.shape[:2]
    coords = []
    blocks = []
    stride = block_size - overlap

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y0, x0 = y, x
            y1 = min(H, y + block_size)
            x1 = min(W, x + block_size)

            if y1 - y0 < block_size and H >= block_size:
                y0 = H - block_size; y1 = H
            if x1 - x0 < block_size and W >= block_size:
                x0 = W - block_size; x1 = W

            if (y0, y1, x0, x1) not in coords:
                coords.append((y0, y1, x0, x1))
                blocks.append(img[y0:y1, x0:x1].copy())

    return blocks, coords

# =====================================================================
# 5. AKNN 核心搜索算法 (纯粹的局部结构探索)
# =====================================================================
def compute_patch_distance(img, y, x, ny, nx, patch_size):
    h, w = img.shape[:2]
    r = patch_size // 2
    y_min, y_max = max(0, y - r), min(h, y + r + 1)
    x_min, x_max = max(0, x - r), min(w, x + r + 1)
    patch_src = img[y_min:y_max, x_min:x_max]

    ny_min = ny - (y - y_min)
    nx_min = nx - (x - x_min)
    ny_max = ny_min + patch_src.shape[0]
    nx_max = nx_min + patch_src.shape[1]

    if ny_min < 0 or nx_min < 0 or ny_max > h or nx_max > w:
        return float('inf')

    patch_target = img[ny_min:ny_max, nx_min:nx_max]
    diff = patch_src - patch_target
    return np.sum(diff * diff)

def initialize_aknn(img, K, patch_size=7, step=2):
    H, W = img.shape[:2]
    r = patch_size // 2
    sigma_s = min(H, W) / 4.0
    ni = np.random.randn(H, W, K, 2)
    vi = np.round(sigma_s * ni).astype(int)

    nn_offsets = np.zeros((H, W, K, 2), dtype=int)
    nn_dists = np.full((H, W, K), float('inf'))

    for y in range(r, H - r, step):
        for x in range(r, W - r, step):
            candidates = []
            for k in range(K):
                dy, dx = vi[y, x, k]
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    dist = compute_patch_distance(img, y, x, ny, nx, patch_size)
                else:
                    dist = float('inf')
                    ny, nx = np.random.randint(0, H), np.random.randint(0, W)
                    dy, dx = ny - y, nx - x
                    dist = compute_patch_distance(img, y, x, ny, nx, patch_size)
                candidates.append((dist, dy, dx))

            candidates.sort(key=lambda x: x[0])
            for k in range(K):
                nn_dists[y, x, k] = candidates[k][0]
                nn_offsets[y, x, k, 0] = candidates[k][1]
                nn_offsets[y, x, k, 1] = candidates[k][2]
    return nn_offsets, nn_dists

def update_best_k(img, y, x, prop_dy, prop_dx, offsets, dists, patch_size, H, W, K):
    ny, nx = y + prop_dy, x + prop_dx
    r = patch_size // 2
    if y - r < 0 or y - r + patch_size > H or x - r < 0 or x - r + patch_size > W: return
    if ny - r < 0 or ny - r + patch_size > H or nx - r < 0 or nx - r + patch_size > W: return
    if (prop_dy != 0 or prop_dx != 0) and (abs(prop_dy) <= r and abs(prop_dx) <= r): return

    patch_src = img[y - r: y - r + patch_size, x - r: x - r + patch_size]
    patch_tgt = img[ny - r: ny - r + patch_size, nx - r: nx - r + patch_size]
    if patch_src.shape[:2] != (patch_size, patch_size) or patch_tgt.shape[:2] != (patch_size, patch_size): return

    new_dist = np.sum((patch_src - patch_tgt) ** 2)
    current_dists = dists[y, x]
    if new_dist >= current_dists[-1]: return

    current_offsets = offsets[y, x]
    for k in range(K):
        if current_offsets[k][0] == prop_dy and current_offsets[k][1] == prop_dx: return

    insert_pos = -1
    for k in range(K):
        if new_dist < current_dists[k]:
            insert_pos = k
            break

    if insert_pos != -1:
        for k in range(K - 1, insert_pos, -1):
            current_dists[k] = current_dists[k - 1]
            current_offsets[k] = current_offsets[k - 1]
        current_dists[insert_pos] = new_dist
        current_offsets[insert_pos] = [prop_dy, prop_dx]

def propagation_step(img, offsets, dists, patch_size, iter_num, step=1):
    H, W = img.shape[:2]
    K = offsets.shape[2]
    r = patch_size // 2
    if iter_num % 2 == 0:
        y_range = range(r + step, H - r, step)
        x_range = range(r + step, W - r, step)
        neighbor_deltas = [(-step, 0), (0, -step)]
    else:
        end_y = r + ((H - r - 1 - r) // step) * step
        end_x = r + ((W - r - 1 - r) // step) * step
        y_range = range(end_y - step, r - 1, -step)
        x_range = range(end_x - step, r - 1, -step)
        neighbor_deltas = [(step, 0), (0, step)]

    for y in y_range:
        for x in x_range:
            for dy_n, dx_n in neighbor_deltas:
                nb_y, nb_x = y + dy_n, x + dx_n
                for k in range(K):
                    prop_dy, prop_dx = offsets[nb_y, nb_x, k]
                    update_best_k(img, y, x, prop_dy, prop_dx, offsets, dists, patch_size, H, W, K)

def random_search_step(img, offsets, dists, patch_size, search_radius, step=1):
    H, W = img.shape[:2]
    K = offsets.shape[2]
    r = patch_size // 2
    for y in range(r, H - r, step):
        for x in range(r, W - r, step):
            for k in range(K):
                best_dy, best_dx = offsets[y, x, k]
                rand_dy = int(round(search_radius * np.random.randn()))
                rand_dx = int(round(search_radius * np.random.randn()))
                update_best_k(img, y, x, best_dy + rand_dy, best_dx + rand_dx, offsets, dists, patch_size, H, W, K)

def run_aknn_pure_python(img, init_offsets, init_dists, iterations, patch_size, step=1):
    offsets = init_offsets.copy()
    dists = init_dists.copy()
    search_radius = min(img.shape[:2])

    for i in range(iterations):
        propagation_step(img, offsets, dists, patch_size, i, step)
        current_radius = max(search_radius * (0.5 ** i), 1)
        random_search_step(img, offsets, dists, patch_size, current_radius, step)
    return offsets, dists

def temporal_local_search(curr_img, prev_img, y, x, patch_size, K_time_max, search_radius):
    """
    在前一帧中搜索与当前帧 (y, x) 位置最相似的 patch。
    支持单通道 (H, W) 和多通道 (H, W, C) 输入 —— SSD 自动对所有通道求和。
    """
    H, W = curr_img.shape[:2]
    r = patch_size // 2
    candidates = []

    min_dy = max(-search_radius, r - y)
    max_dy = min(search_radius, H - r - 1 - y)
    min_dx = max(-search_radius, r - x)
    max_dx = min(search_radius, W - r - 1 - x)

    patch_curr = curr_img[y - r: y + r + 1, x - r: x + r + 1]
    for dy in range(min_dy, max_dy + 1):
        for dx in range(min_dx, max_dx + 1):
            ny, nx = y + dy, x + dx
            patch_prev = prev_img[ny - r: ny + r + 1, nx - r: nx + r + 1]
            diff = patch_curr - patch_prev
            dist = np.sum(diff * diff)
            candidates.append((dist, dy, dx))

    candidates.sort(key=lambda item: item[0])
    return candidates[:K_time_max]

# =====================================================================
# 6. BM3D 引擎 (带有 Kaiser 窗与定长重组)
#    注意：此函数操作单通道 (H, W) 图像
# =====================================================================
def bm3d_1st_stage_video(img_vst_list, offsets_3d, patch_size, step=3):
    curr_img_vst = img_vst_list[0]
    H, W = curr_img_vst.shape
    K_total = offsets_3d.shape[2]
    r = patch_size // 2

    numerator = np.zeros_like(curr_img_vst, dtype=np.float64)
    denominator = np.zeros_like(curr_img_vst, dtype=np.float64)

    sigma_vst = 1.0
    lambda_3d = 2.7 * sigma_vst
    sigma_vst2 = sigma_vst ** 2

    kaiser_1d = np.kaiser(patch_size, 2.0)
    kaiser_2d = np.outer(kaiser_1d, kaiser_1d)

    for y in range(r, H - r, step):
        for x in range(r, W - r, step):
            coords_3d = []
            for k in range(K_total):
                t, dy, dx = offsets_3d[y, x, k]
                if t == -1: continue
                ny, nx = y + dy, x + dx
                if r <= ny <= H - patch_size + r and r <= nx <= W - patch_size + r:
                    coords_3d.append((t, ny, nx))

            K_actual = len(coords_3d)
            if K_actual <= 1: continue

            group_3d = np.zeros((K_actual, patch_size, patch_size), dtype=np.float64)
            for i, (ct, cy, cx) in enumerate(coords_3d):
                group_3d[i] = img_vst_list[ct][cy - r: cy - r + patch_size, cx - r: cx - r + patch_size]

            group_2d_dct = dctn(group_3d, axes=(1, 2), norm='ortho')
            haar_coeffs = pywt.wavedec(group_2d_dct, 'haar', mode='symmetric', axis=0)

            n_nonzero = 0
            for i in range(len(haar_coeffs)):
                haar_coeffs[i][np.abs(haar_coeffs[i]) < lambda_3d] = 0
                n_nonzero += np.sum(haar_coeffs[i] != 0)

            weight = 1.0 / (n_nonzero * sigma_vst2) if n_nonzero > 0 else 1.0 / sigma_vst2

            group_1d_inv = pywt.waverec(haar_coeffs, 'haar', mode='symmetric', axis=0)
            group_1d_inv = group_1d_inv[:K_actual, :, :]
            group_3d_denoised = idctn(group_1d_inv, axes=(1, 2), norm='ortho')

            for i, (ct, cy, cx) in enumerate(coords_3d):
                if ct == 0:
                    numerator[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size] += group_3d_denoised[i] * weight * kaiser_2d
                    denominator[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size] += weight * kaiser_2d

    mask = denominator > 0
    denoised_img = curr_img_vst.copy()
    denoised_img[mask] = numerator[mask] / denominator[mask]
    return denoised_img

# =====================================================================
# 7. 多核并行 Worker —— RAW 版本
#    处理单个 Tile 的时空 BM3D (4 通道 packed Bayer)
# =====================================================================
def process_single_block_video_tile_raw(block_idx,
                                         curr_noisy_vst_block, curr_guide_gr_block,
                                         prev_noisy_vst_block, prev_guide_gr_block,
                                         K_spatial, K_time_max, patch_size, step):
    """
    【极致优化版】：AKNN 仅在单通道 (Gr) 的 guide 上搜索 (算力减少 75%)！
    输入形状: 
      - noisy_vst_block: (H_tile, W_tile, 4) -> 包含所有噪声的 4 通道图
      - guide_gr_block:  (H_tile, W_tile)    -> 仅抽出的一层绿通道向导图
    """
    H, W = curr_noisy_vst_block.shape[:2]

    # 1. 空域搜索 (仅在 1 个 Gr 通道上算距离，速度起飞)
    offsets_spatial, dists_spatial = initialize_aknn(curr_guide_gr_block, K_spatial, patch_size, step)
    offsets_spatial, dists_spatial = run_aknn_pure_python(
        curr_guide_gr_block, offsets_spatial, dists_spatial, iterations=2, patch_size=patch_size, step=step
    )

    K_target_power_of_2 = 8
    offsets_3d = np.full((H, W, K_target_power_of_2, 3), -1, dtype=int)
    local_temporal_stats = {k: 0 for k in range(8)}
    local_total_groups = 0

    # 2. 组装 3D 块
    for cy in range(patch_size // 2, H - patch_size // 2, step):
        for cx in range(patch_size // 2, W - patch_size // 2, step):
            offsets_3d[cy, cx, 0] = [0, 0, 0]
            candidate_pool = []

            if prev_guide_gr_block is not None:
                # 时域搜索：同样仅在 1 个 Gr 通道上算 SSD
                time_cands = temporal_local_search(
                    curr_guide_gr_block, prev_guide_gr_block, cy, cx, patch_size, K_time_max, search_radius=4
                )
                for dist, dy, dx in time_cands:
                    candidate_pool.append((dist, 1, dy, dx))

            for k_s in range(offsets_spatial.shape[2]):
                dy, dx = offsets_spatial[cy, cx, k_s, 0], offsets_spatial[cy, cx, k_s, 1]
                dist = dists_spatial[cy, cx, k_s]
                if not (dy == 0 and dx == 0):
                    candidate_pool.append((dist, 0, dy, dx))

            candidate_pool.sort(key=lambda item: item[0])
            best_7 = candidate_pool[:7]

            valid_count = 1
            time_used = 0
            for dist, t, dy, dx in best_7:
                offsets_3d[cy, cx, valid_count] = [t, dy, dx]
                valid_count += 1
                if t == 1: time_used += 1

            while valid_count < K_target_power_of_2:
                offsets_3d[cy, cx, valid_count] = [0, 0, 0]
                valid_count += 1

            local_temporal_stats[time_used] += 1
            local_total_groups += 1

    # 3. 对 4 个 Bayer 通道逐个运行 BM3D（完美共享仅由 Gr 算出的 offsets_3d）
    denoised_vst_block = np.zeros_like(curr_noisy_vst_block, dtype=np.float64)
    for ch in range(4):
        img_vst_list = [curr_noisy_vst_block[..., ch].astype(np.float64)]
        if prev_noisy_vst_block is not None:
            img_vst_list.append(prev_noisy_vst_block[..., ch].astype(np.float64))

        denoised_vst_block[..., ch] = bm3d_1st_stage_video(img_vst_list, offsets_3d, patch_size, step)

    return block_idx, denoised_vst_block, local_temporal_stats, local_total_groups
# =====================================================================
# 8. 主程序与工具函数
# =====================================================================
def read_raw_mat(path):
    """
    读取 SIDD 格式的 MAT 文件 (h5py)，返回归一化到 [0, 1] 的 float32 RAW。
    同时返回原始最大值，用于逆归一化（若需保存原始量化范围）。
    """
    with h5py.File(path, 'r') as f:
        raw = np.array(f['x']).T.astype(np.float32)

    max_val = float(np.max(raw))
    if max_val > 10.0:
        raw = raw / max_val
    else:
        max_val = 1.0

    return raw, max_val


def process_raw_sequence(mat_folder, out_folder,
                          noise_csv, bayer_csv, scene_id,
                          K_spatial=7, K_time=3, patch_size=7, step=2):
    """
    对一个文件夹下的多帧 RAW MAT 文件进行时空 AKNN-BM3D 降噪。

    参数说明
    --------
    mat_folder : str
        存放 noisy MAT 文件的目录（文件名按字典序视为帧序列）。
        每个 MAT 文件内包含键 'x'，形状为 (H, W) 的 Bayer RAW。
    out_folder : str
        输出降噪 MAT 文件及 CSV 指标的目录。
    noise_csv : str
        SIDD 格式的噪声参数 CSV 路径（包含 beta1_r/g/b, beta2_r/g/b 列）。
    bayer_csv : str
        SIDD 格式的 Bayer 排列 CSV 路径。
    scene_id : str
        场景 ID，用于从 CSV 中查找噪声参数（如 '0065_003_GP_...'）。
        如果输入是合成噪声序列且无需 CSV，可将此参数设为 None，
        此时将使用默认参数 a=0.02, sigma=25/255 对所有通道相同处理。
    K_spatial : int
        空域 AKNN 邻居数。
    K_time : int
        时域候选数上限。
    patch_size : int
        Patch 边长（在 packed 图上的尺寸）。
    step : int
        处理步长。
    """
    mat_files = sorted([f for f in os.listdir(mat_folder)
                        if f.endswith('.mat') and 'NOISY' in f])
    gt_files  = sorted([f for f in os.listdir(mat_folder)
                        if f.endswith('.mat') and 'GT' in f])

    if len(mat_files) == 0:
        # 无 NOISY/GT 前缀时，按所有 mat 文件作为帧序列，不计算指标
        mat_files = sorted([f for f in os.listdir(mat_folder) if f.endswith('.mat')])
        gt_files  = []

    os.makedirs(out_folder, exist_ok=True)
    csv_file_path = os.path.join(out_folder, f"AKNN_VBM3D_RAW_K{K_spatial}_{K_time}.csv")

    csv_file = open(csv_file_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame_Index', 'Filename', 'Time(s)', 'PSNR(dB)', 'SSIM'])

    # 获取噪声参数
    if scene_id is not None:
        # 【修改】：接收 gr_index
        channel_params, bayer_pattern, gr_index = get_noise_and_bayer_info(scene_id, noise_csv, bayer_csv)
        print(f"[Info] 场景 {scene_id} | Bayer: {bayer_pattern.upper()} | Gr 通道索引: {gr_index}")
        # ...
    else:
        a_default, sigma_default = 0.02, 25.0 / 255.0
        channel_params = [(a_default, sigma_default)] * 4
        bayer_pattern = 'rggb'
        gr_index = 1  # 默认 RGGB 的 Gr 是索引 1
        print(f"[Info] 未提供 scene_id，使用默认参数... Gr索引默认为 1")

    prev_packed_vst = None

    # 硬件架构参数：packed 图上的 Tile 尺寸
    # packed 后分辨率减半，83x83 ≈ 原图 166x166 的一半
    TILE_SIZE = 83
    OVERLAP = 19

    for frame_idx, filename in enumerate(mat_files):
        print(f"\n{'='*20} 正在处理帧 {frame_idx:03d}: {filename} {'='*20}")
        noisy_path = os.path.join(mat_folder, filename)
        noisy_raw, max_val = read_raw_mat(noisy_path)

        # 读取对应 GT（用于计算指标）
        gt_raw = None
        if frame_idx < len(gt_files):
            gt_path = os.path.join(mat_folder, gt_files[frame_idx])
            gt_raw, _ = read_raw_mat(gt_path)

        # 1. 打包 Bayer: (H, W) → (H/2, W/2, 4)
        noisy_packed = pack_bayer(noisy_raw)

        # 2. 对 4 个通道分别前向 VST
        curr_packed_vst = np.zeros_like(noisy_packed, dtype=np.float32)
        for ch in range(4):
            a_ch, sigma_ch = channel_params[ch]
            curr_packed_vst[..., ch] = forward_gat(noisy_packed[..., ch], a_ch, sigma_ch)

        # 3. 生成 4 通道联合 guide 图（轻度高斯模糊压制高频噪声）
        curr_guide_vst = np.zeros_like(curr_packed_vst)
        for ch in range(4):
            curr_guide_vst[..., ch] = cv2.GaussianBlur(curr_packed_vst[..., ch], (5, 5), 1.5)

        if prev_packed_vst is not None:
            prev_guide_vst = np.zeros_like(prev_packed_vst)
            for ch in range(4):
                prev_guide_vst[..., ch] = cv2.GaussianBlur(prev_packed_vst[..., ch], (5, 5), 1.5)
        else:
            prev_guide_vst = None

        # 4. 切分 Tile（在 packed 图上操作）
        curr_noisy_blocks, block_coords = split_image_into_grid(curr_packed_vst, TILE_SIZE, OVERLAP)
        curr_guide_blocks, _ = split_image_into_grid(curr_guide_vst, TILE_SIZE, OVERLAP)

        if prev_packed_vst is not None:
            prev_noisy_blocks, _ = split_image_into_grid(prev_packed_vst, TILE_SIZE, OVERLAP)
            prev_guide_blocks, _ = split_image_into_grid(prev_guide_vst, TILE_SIZE, OVERLAP)
        else:
            prev_noisy_blocks = [None] * len(curr_noisy_blocks)
            prev_guide_blocks = [None] * len(curr_noisy_blocks)

        denoised_vst_blocks = [None] * len(curr_noisy_blocks)
        global_temporal_stats = {k: 0 for k in range(8)}
        global_total_groups = 0

        # 5. 多进程并发处理所有 Tile
        t_start = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
            futures = []
            for i in range(len(curr_noisy_blocks)):
                # 【核心修改】：只切片抽出 Gr 通道喂给 Worker！
                # curr_guide_gr_only = curr_guide_blocks[i][..., gr_index]
                # prev_guide_gr_only = prev_guide_blocks[i][..., gr_index] if prev_guide_blocks[i] is not None else None
                
                curr_guide_luma = np.mean(curr_guide_blocks[i], axis=-1)
                
                if prev_guide_blocks[i] is not None:
                    prev_guide_luma = np.mean(prev_guide_blocks[i], axis=-1)
                else:
                    prev_guide_luma = None

                # future = executor.submit(
                #     process_single_block_video_tile_raw,
                #     i, curr_noisy_blocks[i], curr_guide_gr_only,
                #     prev_noisy_blocks[i], prev_guide_gr_only,
                #     K_spatial, K_time, patch_size, step
                # )
                future = executor.submit(
                    process_single_block_video_tile_raw,
                    i, 
                    curr_noisy_blocks[i], 
                    curr_guide_luma,        # 传入融合后的引导层
                    prev_noisy_blocks[i], 
                    prev_guide_luma,       # 传入融合后的引导层（时域）
                    K_spatial, 
                    K_time, 
                    patch_size, 
                    step
                )

                futures.append(future)

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Tiles"):
                block_idx, result_block, local_stats, local_groups = future.result()
                denoised_vst_blocks[block_idx] = result_block
                for k, v in local_stats.items():
                    global_temporal_stats[k] += v
                global_total_groups += local_groups

        duration = time.time() - t_start

        # 打印时空竞争分布
        if prev_packed_vst is not None and global_total_groups > 0:
            print("\n  [硬件级监控] 全局时空大逃杀竞争结果 (胜出时域块数量分布):")
            for k in range(K_time + 1):
                pct = (global_temporal_stats[k] / global_total_groups) * 100
                print(f"    包含 {k} 个时域块的组数: {global_temporal_stats[k]:5d} ({pct:5.1f}%) | {'█' * int(pct/5)}")

        # 6. 将处理好的 Tile 融合回 packed 分辨率
        H_pack, W_pack = curr_packed_vst.shape[:2]
        numerator   = np.zeros((H_pack, W_pack, 4), dtype=np.float64)
        denominator = np.zeros((H_pack, W_pack, 4), dtype=np.float64)

        for i in range(len(curr_noisy_blocks)):
            y0, y1, x0, x1 = block_coords[i]
            numerator[y0:y1, x0:x1, :] += denoised_vst_blocks[i]
            denominator[y0:y1, x0:x1, :] += 1.0

        denoised_packed_vst = (numerator / denominator).astype(np.float32)

        # 7. 逐通道逆 VST → 还原 packed 像素值
        denoised_packed = np.zeros_like(noisy_packed)
        for ch in range(4):
            a_ch, sigma_ch = channel_params[ch]
            denoised_ch = inverse_gat(denoised_packed_vst[..., ch], a_ch, sigma_ch)
            denoised_packed[..., ch] = np.clip(denoised_ch, 0.0, 1.0).astype(np.float32)

        # 8. 解包 Bayer: (H/2, W/2, 4) → (H, W)
        denoised_raw = unpack_bayer(denoised_packed)

        # 9. 计算客观指标
        current_psnr, current_ssim = float('nan'), float('nan')
        if gt_raw is not None:
            current_psnr = psnr(gt_raw, denoised_raw, data_range=1.0)
            current_ssim = ssim(gt_raw, denoised_raw, data_range=1.0)
            print(f"  > 帧处理完成！耗时: {duration:.2f}s | PSNR: {current_psnr:.2f} dB | SSIM: {current_ssim:.4f}")
        else:
            print(f"  > 帧处理完成！耗时: {duration:.2f}s（无 GT，跳过指标计算）")

        csv_writer.writerow([frame_idx, filename, round(duration, 2),
                             round(current_psnr, 4) if not np.isnan(current_psnr) else 'N/A',
                             round(current_ssim, 4) if not np.isnan(current_ssim) else 'N/A'])
        csv_file.flush()

        # 10. 保存降噪结果
        out_mat_path = os.path.join(out_folder, filename.replace('NOISY', 'DENOISED_v8raw'))
        sio.savemat(out_mat_path, {'x': denoised_raw * max_val})
        print(f"  > 降噪 RAW 已保存: {out_mat_path}")

        # 11. 将当前帧的 packed VST 存入 prev，用于下一帧时域搜索
        prev_packed_vst = denoised_packed_vst.copy()

    csv_file.close()
    print(f"\n✅ 所有 RAW 帧已处理完毕！指标已保存至 {csv_file_path}")


# =====================================================================
# 9. CRVD 数据集适配层
#    复用上方所有核心算法，仅替换数据 I/O 和参数加载
# =====================================================================

# CRVD 相机参数（Sony IMX385，与 crvd_raw2png.py 一致）
CRVD_BLACK = 240.0
CRVD_WHITE = 4095.0
CRVD_RANGE = CRVD_WHITE - CRVD_BLACK   # 3855.0

# CRVD Bayer 排列：GBRG
# pack_bayer 通道顺序: ch0=(偶行,偶列)=G, ch1=(偶行,奇列)=B,
#                      ch2=(奇行,偶列)=R, ch3=(奇行,奇列)=G(b)
# → Gr 通道索引 = 0
CRVD_GR_INDEX = 0


def load_crvd_noise_params(
    csv_path: str,
) -> dict:
    """
    读取 crvd_noise_params.csv，返回
        (scene, iso, frame) -> (a_gat, sigma_gat)
    其中：
        a_gat    = a_raw / CRVD_RANGE        (归一化域 Poisson 增益)
        sigma_gat = sqrt(b_raw) / CRVD_RANGE  (归一化域 Gaussian 标准差)
    所有 4 个 Bayer 通道使用相同参数（CRVD CSV 未做通道区分）。
    """
    params = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['scene'], int(row['iso']), int(row['frame']))
            if key not in params:
                a_raw = float(row['a'])
                b_raw = max(float(row['b']), 0.0)
                a_gat     = a_raw / CRVD_RANGE
                sigma_gat = np.sqrt(b_raw) / CRVD_RANGE
                params[key] = (a_gat, sigma_gat)
    return params


def process_crvd_sequence(
    iso_dir:      str,
    noisy_idx:    int,
    scene:        str,
    iso_val:      int,
    noise_params: dict,
    out_dir:      str,
    K_spatial:    int = 7,
    K_time:       int = 3,
    patch_size:   int = 7,
    step:         int = 2,
) -> list[dict]:
    """
    处理 CRVD 一个 (scene, ISO, noisy_idx) 序列（帧 1-7）。
    复用 process_single_block_video_tile_raw 的 40 核 Tile 并行逻辑。

    返回每帧的指标列表，每项含:
        scene, iso, frame, noisy_idx, psnr_noisy, psnr_denoised, time_s
    """
    frame_ids = sorted([
        int(re.search(r'frame(\d+)_clean\.tiff', f).group(1))
        for f in os.listdir(iso_dir)
        if re.search(r'frame(\d+)_clean\.tiff', f)
    ])

    os.makedirs(out_dir, exist_ok=True)
    results = []
    prev_packed_vst = None

    TILE_SIZE = 83
    OVERLAP   = 19

    for frame_id in frame_ids:
        noisy_path = os.path.join(iso_dir, f"frame{frame_id}_noisy{noisy_idx}.tiff")
        clean_path = os.path.join(iso_dir, f"frame{frame_id}_clean.tiff")
        if not os.path.exists(noisy_path) or not os.path.exists(clean_path):
            continue

        key = (scene, iso_val, frame_id)
        if key not in noise_params:
            print(f"[SKIP] 找不到噪声参数: {key}")
            continue
        a_gat, sigma_gat = noise_params[key]
        channel_params = [(a_gat, sigma_gat)] * 4   # 4 通道相同参数

        # ── 读取并归一化 ──
        noisy_raw = cv2.imread(noisy_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        clean_raw = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        noisy_norm = np.clip((noisy_raw - CRVD_BLACK) / CRVD_RANGE, 0.0, 1.0)
        clean_norm = np.clip((clean_raw - CRVD_BLACK) / CRVD_RANGE, 0.0, 1.0)

        psnr_noisy = float(psnr(clean_norm, noisy_norm, data_range=1.0))

        # ── Bayer 打包 + 前向 GAT ──
        noisy_packed = pack_bayer(noisy_norm)
        curr_packed_vst = np.zeros_like(noisy_packed, dtype=np.float32)
        for ch in range(4):
            curr_packed_vst[..., ch] = forward_gat(
                noisy_packed[..., ch], a_gat, sigma_gat)

        # ── Guide 图（轻度高斯模糊）──
        curr_guide_vst = np.zeros_like(curr_packed_vst)
        for ch in range(4):
            curr_guide_vst[..., ch] = cv2.GaussianBlur(
                curr_packed_vst[..., ch], (5, 5), 1.5)

        if prev_packed_vst is not None:
            prev_guide_vst = np.zeros_like(prev_packed_vst)
            for ch in range(4):
                prev_guide_vst[..., ch] = cv2.GaussianBlur(
                    prev_packed_vst[..., ch], (5, 5), 1.5)
        else:
            prev_guide_vst = None

        # ── Tile 切分 ──
        curr_noisy_blocks, block_coords = split_image_into_grid(
            curr_packed_vst, TILE_SIZE, OVERLAP)
        curr_guide_blocks, _ = split_image_into_grid(
            curr_guide_vst, TILE_SIZE, OVERLAP)

        if prev_packed_vst is not None:
            prev_noisy_blocks, _ = split_image_into_grid(prev_packed_vst,   TILE_SIZE, OVERLAP)
            prev_guide_blocks, _ = split_image_into_grid(prev_guide_vst,    TILE_SIZE, OVERLAP)
        else:
            prev_noisy_blocks = [None] * len(curr_noisy_blocks)
            prev_guide_blocks = [None] * len(curr_noisy_blocks)

        denoised_vst_blocks = [None] * len(curr_noisy_blocks)

        # ── 40 核并行处理所有 Tile ──
        t_start = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
            futures = []
            for i in range(len(curr_noisy_blocks)):
                curr_gr = curr_guide_blocks[i][..., CRVD_GR_INDEX]
                prev_gr = (prev_guide_blocks[i][..., CRVD_GR_INDEX]
                           if prev_guide_blocks[i] is not None else None)
                futures.append(executor.submit(
                    process_single_block_video_tile_raw,
                    i, curr_noisy_blocks[i], curr_gr,
                    prev_noisy_blocks[i], prev_gr,
                    K_spatial, K_time, patch_size, step,
                ))
            for future in concurrent.futures.as_completed(futures):
                blk_idx, result_blk, _, _ = future.result()
                denoised_vst_blocks[blk_idx] = result_blk
        t_elapsed = time.time() - t_start

        # ── Tile 融合 ──
        H_pack, W_pack = curr_packed_vst.shape[:2]
        numerator   = np.zeros((H_pack, W_pack, 4), dtype=np.float64)
        denominator = np.zeros((H_pack, W_pack, 4), dtype=np.float64)
        for i, (y0, y1, x0, x1) in enumerate(block_coords):
            numerator[y0:y1, x0:x1, :]   += denoised_vst_blocks[i]
            denominator[y0:y1, x0:x1, :] += 1.0
        denoised_packed_vst = (numerator / denominator).astype(np.float32)

        # ── 逆 GAT + 解包 ──
        denoised_packed = np.zeros_like(noisy_packed)
        for ch in range(4):
            denoised_ch = inverse_gat(denoised_packed_vst[..., ch], a_gat, sigma_gat)
            denoised_packed[..., ch] = np.clip(denoised_ch, 0.0, 1.0).astype(np.float32)
        denoised_norm = unpack_bayer(denoised_packed)

        psnr_denoised = float(psnr(clean_norm, denoised_norm, data_range=1.0))
        print(f"  [{scene}/{os.path.basename(iso_dir)}/frame{frame_id}/n{noisy_idx}] "
              f"PSNR: {psnr_noisy:.2f} → {psnr_denoised:.2f} dB  ({t_elapsed:.1f}s)")

        # ── 保存降噪 TIFF（反归一化到 uint16）──
        denoised_uint16 = np.clip(
            denoised_norm * CRVD_RANGE + CRVD_BLACK, 0, CRVD_WHITE
        ).astype(np.uint16)
        out_tiff = os.path.join(out_dir,
                                f"frame{frame_id}_noisy{noisy_idx}_denoised.tiff")
        cv2.imwrite(out_tiff, denoised_uint16)

        results.append({
            'scene':            scene,
            'iso':              iso_val,
            'frame':            frame_id,
            'noisy_idx':        noisy_idx,
            'psnr_noisy_dB':    round(psnr_noisy,    4),
            'psnr_denoised_dB': round(psnr_denoised, 4),
            'time_s':           round(t_elapsed,     1),
            'a_gat':            round(a_gat,          8),
            'sigma_gat':        round(sigma_gat,      8),
        })

        prev_packed_vst = denoised_packed_vst.copy()

    return results


def run_crvd(
    crvd_root:   str = 'data/CRVD/noisy',
    out_root:    str = 'out/results/CRVD/AKNN_v8',
    params_csv:  str = 'out/results/crvd_noise_params.csv',
    K_spatial:   int = 7,
    K_time:      int = 3,
    patch_size:  int = 7,
    step:        int = 2,
) -> None:
    """
    遍历 CRVD 所有 (scene, ISO, noisy_idx)，调用 process_crvd_sequence，
    将 PSNR 指标汇总写入 CSV。
    """
    print(f"加载噪声参数: {params_csv}")
    noise_params = load_crvd_noise_params(params_csv)
    print(f"  已读取 {len(noise_params)} 组 (scene, iso, frame) 参数\n")

    all_rows = []

    scene_dirs = sorted(glob.glob(os.path.join(crvd_root, 'scene*')))
    for scene_dir in scene_dirs:
        scene = os.path.basename(scene_dir)
        for iso_dir in sorted(glob.glob(os.path.join(scene_dir, 'ISO*'))):
            iso_name = os.path.basename(iso_dir)
            iso_val  = int(re.sub(r'\D', '', iso_name))

            # 推断当前 ISO 目录下的 noisy_idx 范围
            noisy_indices = sorted(set(
                int(re.search(r'frame\d+_noisy(\d+)\.tiff', f).group(1))
                for f in os.listdir(iso_dir)
                if re.search(r'frame\d+_noisy(\d+)\.tiff', f)
            ))

            for noisy_idx in noisy_indices:
                print(f"\n{'='*20} {scene}/{iso_name}/noisy{noisy_idx} {'='*20}")
                out_dir = os.path.join(out_root, scene, iso_name)
                rows = process_crvd_sequence(
                    iso_dir      = iso_dir,
                    noisy_idx    = noisy_idx,
                    scene        = scene,
                    iso_val      = iso_val,
                    noise_params = noise_params,
                    out_dir      = out_dir,
                    K_spatial    = K_spatial,
                    K_time       = K_time,
                    patch_size   = patch_size,
                    step         = step,
                )
                all_rows.extend(rows)

    # ── 写 CSV ──
    os.makedirs(out_root, exist_ok=True)
    csv_out = os.path.join(out_root, 'results.csv')
    fieldnames = ['scene', 'iso', 'frame', 'noisy_idx',
                  'psnr_noisy_dB', 'psnr_denoised_dB',
                  'time_s', 'a_gat', 'sigma_gat']
    all_rows.sort(key=lambda r: (r['scene'], r['iso'], r['frame'], r['noisy_idx']))
    with open(csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n结果已保存：{csv_out}  ({len(all_rows)} 行)")

    # ── 汇总 ──
    from collections import defaultdict
    stats = defaultdict(list)
    for r in all_rows:
        stats[r['iso']].append((r['psnr_noisy_dB'], r['psnr_denoised_dB']))
    print(f"\n{'ISO':>8}  {'PSNR_noisy(dB)':>16}  {'PSNR_denoised(dB)':>18}  {'N':>5}")
    print('-' * 55)
    for iso in sorted(stats):
        arr = np.array(stats[iso])
        print(f"{iso:>8}  {np.mean(arr[:,0]):>16.3f}  "
              f"{np.mean(arr[:,1]):>18.3f}  {len(arr):>5}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='AKNN-BM3D v8 RAW 降噪')
    sub = parser.add_subparsers(dest='mode', required=True)

    # ── SIDD 模式（原有功能保留）──
    p_sidd = sub.add_parser('sidd', help='SIDD 数据集（MAT 格式）')
    p_sidd.add_argument('--dataset',    default='data/SIDD_small_RAW/')
    p_sidd.add_argument('--scene',      default='0065_003_GP_10000_08460_4400_N')
    p_sidd.add_argument('--K_spatial',  type=int, default=7)
    p_sidd.add_argument('--K_time',     type=int, default=3)
    p_sidd.add_argument('--patch_size', type=int, default=7)
    p_sidd.add_argument('--step',       type=int, default=2)

    # ── CRVD 模式（新功能）──
    p_crvd = sub.add_parser('crvd', help='CRVD 数据集（TIFF 格式）')
    p_crvd.add_argument('--root',       default='data/CRVD/noisy')
    p_crvd.add_argument('--out',        default='out/results/CRVD/AKNN_v8')
    p_crvd.add_argument('--params',     default='out/results/crvd_noise_params.csv')
    p_crvd.add_argument('--K_spatial',  type=int, default=7)
    p_crvd.add_argument('--K_time',     type=int, default=3)
    p_crvd.add_argument('--patch_size', type=int, default=7)
    p_crvd.add_argument('--step',       type=int, default=2)

    args = parser.parse_args()

    if args.mode == 'sidd':
        dataset_path = args.dataset
        scene_dir    = Path(dataset_path) / args.scene
        process_raw_sequence(
            mat_folder  = str(scene_dir),
            out_folder  = str(scene_dir / 'res_v8_raw'),
            noise_csv   = str(Path(dataset_path) / 'noise_level_functions.csv'),
            bayer_csv   = str(Path(dataset_path) / 'bayer_patterns.csv'),
            scene_id    = args.scene,
            K_spatial   = args.K_spatial,
            K_time      = args.K_time,
            patch_size  = args.patch_size,
            step        = args.step,
        )
    else:
        run_crvd(
            crvd_root  = args.root,
            out_root   = args.out,
            params_csv = args.params,
            K_spatial  = args.K_spatial,
            K_time     = args.K_time,
            patch_size = args.patch_size,
            step       = args.step,
        )
