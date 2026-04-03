import os
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
# 2. 硬件架构映射：Tile-based 切分与重组 (Core + Halo)
# =====================================================================
def split_image_into_grid(img, block_size=166, overlap=38):
    """
    完美模拟硬件的 Core Tile + Halo Margin 架构。
    当 block_size=166, overlap=38 时，每次滑动步长(Stride)恰好是 128。
    这意味着核心有效区是 128x128，外围的 19 像素是提供给边界的 Halo 搜索视野。
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
            
            # 边界对齐
            if y1 - y0 < block_size and H >= block_size:
                y0 = H - block_size; y1 = H
            if x1 - x0 < block_size and W >= block_size:
                x0 = W - block_size; x1 = W

            if (y0, y1, x0, x1) not in coords:
                coords.append((y0, y1, x0, x1))
                blocks.append(img[y0:y1, x0:x1].copy())

    return blocks, coords

# =====================================================================
# 3. AKNN 核心搜索算法 (纯粹的局部结构探索)
# =====================================================================
def compute_patch_distance(img, y, x, ny, nx, patch_size):
    h, w = img.shape
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

def initialize_aknn(img, K, patch_size=7, step=1):
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
    if patch_src.shape != (patch_size, patch_size) or patch_tgt.shape != (patch_size, patch_size): return

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
    
    # 并在多进程中避免打印太多垃圾信息
    for i in range(iterations):
        propagation_step(img, offsets, dists, patch_size, i, step)
        current_radius = max(search_radius * (0.5 ** i), 1)
        random_search_step(img, offsets, dists, patch_size, current_radius, step)
    return offsets, dists

def temporal_local_search(curr_img, prev_img, y, x, patch_size, K_time_max, search_radius):
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
# 4. BM3D 引擎 (带有 Kaiser 窗与定长重组)
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

    # 消除边界网格效应的 Kaiser 窗
    kaiser_1d = np.kaiser(patch_size, 2.0)
    kaiser_2d = np.outer(kaiser_1d, kaiser_1d)

    for y in range(r, H - r, step):
        for x in range(r, W - r, step):
            coords_3d = []  # 修正：空列表起手，因为 offsets 第 0 位已经是原块
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
                if ct == 0:  # 仅把去噪结果加回到当前帧
                    numerator[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size] += group_3d_denoised[i] * weight * kaiser_2d
                    denominator[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size] += weight * kaiser_2d

    mask = denominator > 0
    denoised_img = curr_img_vst.copy()
    denoised_img[mask] = numerator[mask] / denominator[mask]
    return denoised_img

# =====================================================================
# 5. 多核并行 Worker (执行单个 Tile 的时空大逃杀)
# =====================================================================
def process_single_block_video_tile(block_idx, curr_noisy_vst_block, curr_guide_vst_block,
                                    prev_noisy_vst_block, prev_guide_vst_block,
                                    K_spatial, K_time_max, patch_size, step):
    H, W = curr_noisy_vst_block.shape

    # 1. 空域搜索 (仅在 guide 图上跑)
    offsets_spatial, dists_spatial = initialize_aknn(curr_guide_vst_block, K_spatial, patch_size, step)
    offsets_spatial, dists_spatial = run_aknn_pure_python(
        curr_guide_vst_block, offsets_spatial, dists_spatial, iterations=2, patch_size=patch_size, step=step
    )

    K_target_power_of_2 = 8
    offsets_3d = np.full((H, W, K_target_power_of_2, 3), -1, dtype=int)
    local_temporal_stats = {k: 0 for k in range(8)}
    local_total_groups = 0

    # 2. 组装 3D 块 (时空大逃杀排序池)
    for cy in range(patch_size // 2, H - patch_size // 2, step):
        for cx in range(patch_size // 2, W - patch_size // 2, step):
            offsets_3d[cy, cx, 0] = [0, 0, 0] # 队长
            candidate_pool = []

            # 【修复】：时域选用 guide_vst 算公平距离
            if prev_guide_vst_block is not None:
                time_cands = temporal_local_search(
                    curr_guide_vst_block, prev_guide_vst_block, cy, cx, patch_size, K_time_max, search_radius=4
                )
                for dist, dy, dx in time_cands:
                    candidate_pool.append((dist, 1, dy, dx))

            # 空域选手进池子
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

    # 3. 传入跨帧 BM3D 引擎 (注意：必须传带原始噪声的 block！)
    img_vst_list = [curr_noisy_vst_block]
    if prev_noisy_vst_block is not None:
        img_vst_list.append(prev_noisy_vst_block)
        
    denoised_vst_block = bm3d_1st_stage_video(img_vst_list, offsets_3d, patch_size, step)
    
    return block_idx, denoised_vst_block, local_temporal_stats, local_total_groups

# =====================================================================
# 6. 主程序与工具函数
# =====================================================================
def read_png_to_yuv(path, normalize=True):
    img = cv2.imread(path)
    if img is None: return None, None, None, None
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = yuv[:, :, 0].astype(np.float32) / 255.0 if normalize else yuv[:, :, 0]
    cr = yuv[:, :, 1].astype(np.float32) / 255.0 if normalize else yuv[:, :, 1]
    cb = yuv[:, :, 2].astype(np.float32) / 255.0 if normalize else yuv[:, :, 2]
    return y, cb, cr, img

def add_poisson_gaussian_noise(img_clean, a=0.1, sigma_norm=25/255, seed=None):
    rng = np.random.default_rng(seed)
    photon_counts = np.maximum(img_clean / a, 1e-10)
    noisy_poisson = rng.poisson(photon_counts) * a
    noisy_gaussian = rng.normal(0, sigma_norm, img_clean.shape)
    return np.clip(noisy_poisson + noisy_gaussian, 0.0, 1.0).astype(np.float32)

def process_video_sequence(png_folder, out_folder, K_spatial=7, K_time=3, patch_size=7, step=2):
    png_files = sorted([f for f in os.listdir(png_folder) if f.endswith('.png')])
    os.makedirs(out_folder, exist_ok=True)
    csv_file_path = os.path.join(out_folder, f"AKNN_VBM3D_Golden_Model_K{K_spatial}_{K_time}.csv")

    csv_file = open(csv_file_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame_Index', 'Filename', 'Time(s)', 'PSNR(dB)', 'SSIM'])

    prev_y_vst = None  
    
    # 【架构敲定】：采用 166x166 的 Tile 包含 38 像素重叠 (模拟 128 Core + Halo)
    TILE_SIZE = 166
    OVERLAP = 38

    for frame_idx, filename in enumerate(png_files):
        print(f"\n{'='*20} 正在处理帧 {frame_idx:03d}: {filename} {'='*20}")
        img_path = os.path.join(png_folder, filename)
        y, cb, cr, img_bgr = read_png_to_yuv(img_path)
        
        # 种子锁定为 42 以便于学术复现
        y_noisy = add_poisson_gaussian_noise(y, a=0.02, sigma_norm=25 / 255, seed=42)

        # 1. 全局 VST 变换与向导图生成 (当前帧和前一帧都必须生成！)
        curr_y_vst = forward_gat(y_noisy, a=0.02, sigma=25 / 255)
        curr_guide_vst = cv2.GaussianBlur(curr_y_vst, (5, 5), 1.5)
        
        if prev_y_vst is not None:
            prev_guide_vst = cv2.GaussianBlur(prev_y_vst, (5, 5), 1.5)
        else:
            prev_guide_vst = None

        # 2. 全图切分成带 Overlap 的 Tile 网格
        curr_noisy_blocks, block_coords = split_image_into_grid(curr_y_vst, TILE_SIZE, OVERLAP)
        curr_guide_blocks, _ = split_image_into_grid(curr_guide_vst, TILE_SIZE, OVERLAP)
        
        if prev_y_vst is not None:
            prev_noisy_blocks, _ = split_image_into_grid(prev_y_vst, TILE_SIZE, OVERLAP)
            prev_guide_blocks, _ = split_image_into_grid(prev_guide_vst, TILE_SIZE, OVERLAP)
        else:
            prev_noisy_blocks = [None] * len(curr_noisy_blocks)
            prev_guide_blocks = [None] * len(curr_noisy_blocks)

        denoised_vst_blocks = [None] * len(curr_noisy_blocks)
        global_temporal_stats = {k: 0 for k in range(8)}
        global_total_groups = 0

        # 3. 多进程并发处理所有的 Tile (榨干 CPU 性能)
        t_start = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor:
            futures = []
            for i in range(len(curr_noisy_blocks)):
                future = executor.submit(
                    process_single_block_video_tile,
                    i, curr_noisy_blocks[i], curr_guide_blocks[i],
                    prev_noisy_blocks[i], prev_guide_blocks[i],
                    K_spatial, K_time, patch_size, step
                )
                futures.append(future)

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Tiles"):
                block_idx, result_block, local_stats, local_groups = future.result()
                denoised_vst_blocks[block_idx] = result_block
                for k, v in local_stats.items():
                    global_temporal_stats[k] += v
                global_total_groups += local_groups

        duration = time.time() - t_start

        # 打印硬件级时空竞争分布图表
        if prev_y_vst is not None and global_total_groups > 0:
            print("\n  [硬件级监控] 全局时空大逃杀竞争结果 (胜出时域块数量分布):")
            for k in range(K_time + 1):
                pct = (global_temporal_stats[k] / global_total_groups) * 100
                print(f"    包含 {k} 个时域块的组数: {global_temporal_stats[k]:5d} ({pct:5.1f}%) | {'█' * int(pct/5)}")

        # 4. 将处理好的 Tile 完美融合回全图分辨率
        H, W = curr_y_vst.shape
        numerator = np.zeros((H, W), dtype=np.float32)
        denominator = np.zeros((H, W), dtype=np.float32)

        for i in range(len(curr_noisy_blocks)):
            y0, y1, x0, x1 = block_coords[i]
            numerator[y0:y1, x0:x1] += denoised_vst_blocks[i]
            denominator[y0:y1, x0:x1] += 1.0

        denoised_y_vst = numerator / denominator

        # 5. 逆 VST 变换与客观指标计算
        denoised_y = inverse_gat(denoised_y_vst, a=0.02, sigma=25 / 255)
        denoised_y = np.clip(denoised_y, 0.0, 1.0)
        
        current_psnr = psnr(y, denoised_y, data_range=1.0)
        current_ssim = ssim(y, denoised_y, data_range=1.0)
        print(f"  > 帧处理完成！耗时: {duration:.2f}s | PSNR: {current_psnr:.2f} dB | SSIM: {current_ssim:.4f}")
        
        csv_writer.writerow([frame_idx, filename, round(duration, 2), round(current_psnr, 4), round(current_ssim, 4)])
        csv_file.flush() 

        # 6. 【核心修复】将处理完的干净图像存入 prev_y_vst，用于下一帧降噪
        prev_y_vst = denoised_y_vst.copy()

    csv_file.close()
    print(f"\n✅ 所有视频帧已处理完毕！指标已安全保存至 {csv_file_path}")

if __name__ == "__main__":
    png_folder = "data/DAVIS/car-turn/720p"  # 替换为你的输入文件夹路径
    out_folder = "data/DAVIS/car-turn/res_v8"  # 替换为你的输出文件夹路径
    
    # 开始执行核心视频流水线
    process_video_sequence(png_folder, out_folder, K_spatial=7, K_time=3, patch_size=7, step=2)