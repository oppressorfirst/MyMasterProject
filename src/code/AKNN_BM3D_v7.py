import os
import cv2
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.fft import dctn, idctn
import pywt
import sys
import csv


def forward_gat(z, a, sigma):
    """
    广义 Anscombe 变换 (GAT)
    将泊松-高斯混合噪声转化为近似标准差为 1 的高斯白噪声。
    """
    return 2.0 * np.sqrt(np.maximum(z / a + 3.0 / 8.0 + (sigma ** 2) / (a ** 2), 0))

def inverse_gat(D, a, sigma):
    """
    GAT 的渐近逆变换 (Asymptotic Inverse)
    将去噪后的高斯域信号映射回原始的泊松-高斯域。
    """
    return a * ((D / 2.0) ** 2 - 1.0 / 8.0 - (sigma ** 2) / (a ** 2))

def split_image_into_4_blocks(img, overlap=39):
    """
    将图像分成上下左右 4 块，包含指定的像素重叠。
    返回: 4个子图构成的列表，以及它们在原图中的切片坐标。
    """
    H, W = img.shape[:2]
    mid_H, mid_W = H // 2, W // 2

    # 定义四个块的边界 (y_start, y_end, x_start, x_end)
    coords = [
        (0, mid_H + overlap, 0, mid_W + overlap),  # Top-Left
        (0, mid_H + overlap, mid_W - overlap, W),  # Top-Right
        (mid_H - overlap, H, 0, mid_W + overlap),  # Bottom-Left
        (mid_H - overlap, H, mid_W - overlap, W)  # Bottom-Right
    ]

    blocks = []
    for (y0, y1, x0, x1) in coords:
        blocks.append(img[y0:y1, x0:x1].copy())

    return blocks, coords





def process_single_block(block_idx, noisy_vst_block, guide_vst_block, K, patch_size, process_step):
    """
    包装单个块的处理流程（在 VST 域上操作）。
    """
    print(f"\n--- [Worker {block_idx}] 开始处理 ---")

    # 注意：此时传入 AKNN 和 BM3D 的都是 VST 变换后的图像
    offsets, dists = initialize_aknn(guide_vst_block, K, patch_size, step=process_step)

    # AKNN 的随机搜索半径逻辑保持不变
    final_offsets, final_dists = run_aknn_pure_python(
        guide_vst_block, offsets, dists, 2, patch_size, sigma_norm=1.0, step=process_step
    )

    # 运行 VST 域专用的 BM3D
    denoised_vst_block = bm3d_1st_stage_vst_offsets(
        img_vst=noisy_vst_block,
        offsets=final_offsets,
        patch_size=patch_size,
        step=process_step
    )

    print(f"--- [Worker {block_idx}] 处理完成 ---")
    return block_idx, denoised_vst_block

def visualize_block_by_index(img, block_idx, offsets, patch_size, step):
    """
    通过一维的块索引 (0 到 total_blocks-1) 来进行可视化
    自动计算出合法的 (y, x) 坐标
    """
    H, W = img.shape[:2]
    r = patch_size // 2

    # 提前计算出合法的 y 和 x 坐标列表
    y_coords = list(range(r, H - r, step))
    x_coords = list(range(r, W - r, step))

    num_y = len(y_coords)
    num_x = len(x_coords)
    total_blocks = num_y * num_x

    # 越界检查
    if block_idx < 0 or block_idx >= total_blocks:
        print(f"错误: 索引 {block_idx} 越界。当前网格(step={step})总共有 {total_blocks} 个块。")
        return

    # 将 1D 索引转换为 2D 的网格索引
    grid_y_idx = block_idx // num_x
    grid_x_idx = block_idx % num_x

    # 获取实际的图像 (y, x) 坐标
    y = y_coords[grid_y_idx]
    x = x_coords[grid_x_idx]

    print(f"Visualizing Block {block_idx} / {total_blocks} -> Mapped to Pixel (y={y}, x={x})")

    # 调用你原有的可视化函数
    visualize_pixel_and_candidates(img, y, x, offsets, patch_size)


def visualize_pixel_and_candidates(img, y0, x0, offsets, patch_size):
    """
    可视化某个像素的 K 个候选
    红框：中心 patch
    蓝框：候选 patch
    """
    H, W = img.shape
    K = offsets.shape[2]
    r = patch_size // 2

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Pixel ({y0}, {x0}) and its {K} candidates")
    ax.axis('off')

    # 画中心像素 patch（红色）
    red_rect = patches.Rectangle(
        (x0 - r, y0 - r), patch_size, patch_size,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(red_rect)

    # 画候选 patch（蓝色）
    for k in range(K):
        dy, dx = offsets[y0, x0, k]
        ny, nx = y0 + dy, x0 + dx

        if 0 <= ny < H and 0 <= nx < W:
            blue_rect = patches.Rectangle(
                (nx - r, ny - r), patch_size, patch_size,
                linewidth=1.5, edgecolor='blue', facecolor='none'
            )
            ax.add_patch(blue_rect)

            # 标号（可选）
            ax.text(nx, ny, f"{k}", color='blue', fontsize=10)

    plt.show()

def compute_patch_distance(img, y, x, ny, nx, patch_size):
    """
    辅助函数：计算两个 Patch 之间的距离 (Sum of Squared Differences)。
    这里为了演示逻辑简单实现，实际应用中可以使用积分图或卷积加速。

    参数:
    img: 图像数组 (H, W, C)
    y, x: 源像素坐标
    ny, nx: 目标(邻居)像素坐标
    patch_size: Patch 的边长 (例如 7)
    """
    h, w= img.shape
    r = patch_size // 2

    # 确定 Patch 的范围，注意处理图像边界
    y_min, y_max = max(0, y - r), min(h, y + r + 1)
    x_min, x_max = max(0, x - r), min(w, x + r + 1)

    # 对应的邻居 Patch 范围
    # 注意：如果源 Patch 在边界被截断，目标 Patch 也应取相同大小的区域以进行比较
    # 这里简化处理：只计算有效重叠区域，或假设 Patch 是完整的。
    # 为了代码健壮性，这里取对应偏移后的切片：

    patch_src = img[y_min:y_max, x_min:x_max]

    # 计算目标区域的起始点
    ny_min = ny - (y - y_min)
    nx_min = nx - (x - x_min)
    ny_max = ny_min + patch_src.shape[0]
    nx_max = nx_min + patch_src.shape[1]

    # 检查目标区域是否越界
    if ny_min < 0 or nx_min < 0 or ny_max > h or nx_max > w:
        return float('inf')  # 越界视为无穷大距离

    patch_target = img[ny_min:ny_max, nx_min:nx_max]

    # 计算 SSD (Sum of Squared Differences)
    diff = patch_src - patch_target
    dist = np.sum(diff * diff)
    return dist


def initialize_aknn(img, K, patch_size=7, step=1):
    H, W = img.shape[:2]
    r = patch_size // 2
    sigma_s = W / 3.0
    ni = np.random.randn(H, W, K, 2)
    vi = np.round(sigma_s * ni).astype(int)

    nn_offsets = np.zeros((H, W, K, 2), dtype=int)
    nn_dists = np.full((H, W, K), float('inf'))

    print(f"Initializing AKNN for image {H}x{W} with K={K}, Step={step}...")

    # 【关键修改】：起点设为 r，步长设为 step，与 BM3D 完美对齐
    for y in tqdm(range(r, H - r, step), desc="Init AKNN", dynamic_ncols=True):
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


# --- 1. 核心辅助函数：维护优先队列 ---
def update_best_k(img, y, x, prop_dy, prop_dx, offsets, dists, patch_size, H, W, K):
    ny, nx = y + prop_dy, x + prop_dx
    r = patch_size // 2

    # 1. 越界检查 (注意修正了奇偶数兼容的写法)
    if y - r < 0 or y - r + patch_size > H or x - r < 0 or x - r + patch_size > W:
        return
    if ny - r < 0 or ny - r + patch_size > H or nx - r < 0 or nx - r + patch_size > W:
        return

    if (prop_dy != 0 or prop_dx != 0) and (abs(prop_dy) <= r and abs(prop_dx) <= r):
        return


    # 2. 提取 Patch
    patch_src = img[y - r: y - r + patch_size, x - r: x - r + patch_size]
    patch_tgt = img[ny - r: ny - r + patch_size, nx - r: nx - r + patch_size]

    if patch_src.shape != (patch_size, patch_size) or patch_tgt.shape != (patch_size, patch_size):
        return

    # 3. 使用魔改后的 DCT 距离替换普通的 SSD！
    diff = patch_src - patch_tgt
    new_dist = np.sum(diff * diff)

    # 4. 检查是否值得插入
    current_dists = dists[y, x]
    if new_dist >= current_dists[-1]:
        return

    current_offsets = offsets[y, x]
    for k in range(K):
        if current_offsets[k][0] == prop_dy and current_offsets[k][1] == prop_dx:
            return

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
    print(f"  > Propagation (Direction: {'Scanline' if iter_num % 2 == 0 else 'Reverse'}, Step: {step})...")

    if iter_num % 2 == 0:
        # 正向：跳过最外层，每次加 step
        y_range = range(r + step, H - r, step)
        x_range = range(r + step, W - r, step)
        neighbor_deltas = [(-step, 0), (0, -step)] # 看左边和上边 step 距离的邻居
    else:
        # 反向：计算出正向网格的最后一个点，确保网格点对齐
        end_y = r + ((H - r - 1 - r) // step) * step
        end_x = r + ((W - r - 1 - r) // step) * step
        y_range = range(end_y - step, r - 1, -step)
        x_range = range(end_x - step, r - 1, -step)
        neighbor_deltas = [(step, 0), (0, step)] # 看右边和下边 step 距离的邻居

    for y in y_range:
        for x in x_range:
            for dy_n, dx_n in neighbor_deltas:
                nb_y, nb_x = y + dy_n, x + dx_n
                # 直接获取对应 step 邻居的偏移量
                nb_offsets = offsets[nb_y, nb_x]
                for k in range(K):
                    prop_dy, prop_dx = nb_offsets[k]
                    update_best_k(img, y, x, prop_dy, prop_dx,
                                  offsets, dists, patch_size, H, W, K)


# --- 3. 随机搜索步骤 (Random Search) ---
def random_search_step(img, offsets, dists, patch_size, search_radius, step=1):
    H, W = img.shape[:2]
    K = offsets.shape[2]
    r = patch_size // 2
    print(f"  > Random Search (Radius: {search_radius:.2f}, Step: {step})...")

    # 【关键修改】：按 step 遍历网格
    for y in range(r, H - r, step):
        for x in range(r, W - r, step):
            for k in range(K):
                best_dy, best_dx = offsets[y, x, k]
                rand_dy = int(round(search_radius * np.random.randn()))
                rand_dx = int(round(search_radius * np.random.randn()))
                search_dy = best_dy + rand_dy
                search_dx = best_dx + rand_dx

                update_best_k(img, y, x, search_dy, search_dx,
                              offsets, dists, patch_size, H, W, K)



# --- 4. 主程序：把所有步骤串起来 ---
def run_aknn_pure_python(img, init_offsets, init_dists, iterations, patch_size, sigma_norm, step=1):
    H, W = img.shape[:2]
    offsets = init_offsets.copy()
    dists = init_dists.copy()
    search_radius = W

    print(f"Starting AKNN Loop ({iterations} iterations, Step={step})...")

    pbar = tqdm(
        range(iterations),
        desc="AKNN Iter",
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=0.1,
        position=0
    )

    for i in pbar:
        t0 = time.time()

        # --- propagation ---
        propagation_step(img, offsets, dists, patch_size, i, step)

        # --- radius decay ---
        current_radius = search_radius * (0.5 ** i)
        if current_radius < 1:
            current_radius = 1

        # --- random search ---
        random_search_step(img, offsets, dists, patch_size, current_radius, step)

        t1 = time.time()
        iter_time = t1 - t0

        # ⭐ 核心改进：用 set_postfix 替代 tqdm.write
        pbar.set_postfix({
            "iter_time": f"{iter_time:.2f}s",
            "radius": int(current_radius),
            "step": step,
            "patch": patch_size
        })

    return offsets, dists


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


def bm3d_1st_stage_vst_offsets(img_vst, offsets, patch_size, step=3):
    """
    接收 AKNN offsets 的 VST 域 BM3D 降噪。
    由于输入图像经过了 VST，此时全局噪声标准差 sigma_vst 恒等于 1.0。
    """
    H, W = img_vst.shape
    K_offsets = offsets.shape[2]
    r = patch_size // 2

    numerator = np.zeros_like(img_vst, dtype=np.float64)
    denominator = np.zeros_like(img_vst, dtype=np.float64)

    print("\nStarting BM3D 1st Stage on VST domain...")

    # 【核心改进】：VST 域的全局噪声标准差被稳定为 1.0
    sigma_vst = 1.0
    lambda_3d = 2.7 * sigma_vst
    sigma_vst2 = sigma_vst ** 2

    for y in trange(r, H - r, step, desc="BM3D 3D Transform"):
        for x in range(r, W - r, step):
            coords = [(y, x)]

            for k in range(K_offsets):
                dy, dx = offsets[y, x, k]
                ny, nx = y + dy, x + dx
                if r <= ny <= H - patch_size + r and r <= nx <= W - patch_size + r:
                    coords.append((ny, nx))

            K_actual = len(coords)
            if K_actual <= 1:
                continue

            group_3d = np.zeros((K_actual, patch_size, patch_size), dtype=np.float64)
            for i, (cy, cx) in enumerate(coords):
                group_3d[i] = img_vst[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size]

            # 3. 混合 3D 变换
            group_2d_dct = dctn(group_3d, axes=(1, 2), norm='ortho')
            haar_coeffs = pywt.wavedec(group_2d_dct, 'haar', mode='symmetric', axis=0)

            # 4. 全局硬阈值截断 (使用固定的 lambda_3d)
            n_nonzero = 0
            for i in range(len(haar_coeffs)):
                haar_coeffs[i][np.abs(haar_coeffs[i]) < lambda_3d] = 0
                n_nonzero += np.sum(haar_coeffs[i] != 0)

            # 5. 计算聚合权重 (使用固定的 sigma_vst2)
            if n_nonzero > 0:
                weight = 1.0 / (n_nonzero * sigma_vst2)
            else:
                weight = 1.0 / sigma_vst2

            # 6. 逆向混合变换
            group_1d_inv = pywt.waverec(haar_coeffs, 'haar', mode='symmetric', axis=0)
            group_1d_inv = group_1d_inv[:K_actual, :, :]
            group_3d_denoised = idctn(group_1d_inv, axes=(1, 2), norm='ortho')

            # 7. 聚合
            for i, (cy, cx) in enumerate(coords):
                numerator[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size] += group_3d_denoised[i] * weight
                denominator[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size] += weight

    mask = denominator > 0
    denoised_img = img_vst.copy()
    denoised_img[mask] = numerator[mask] / denominator[mask]

    return denoised_img


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

    cv2.imwrite(os.path.join(img_save_dir, f"noisy_color_{idx:03d}.png"), noisy_bgr)
    cv2.imwrite(os.path.join(img_save_dir, f"kaiser_denoised_color_{idx:03d}.png"), denoised_bgr)
    cv2.imwrite(os.path.join(img_save_dir, f"AKNN_VBM3D_kaiser_step_2_{idx:03d}_ALL.png"), final_canvas)


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

    # 按距离排序，纯粹按实力说话
    candidates.sort(key=lambda item: item[0])
    return candidates[:K_time_max]


def bm3d_1st_stage_video(img_vst_list, offsets_3d, patch_size, step=3):
    """
    接收跨帧 offsets 的视频域 BM3D 降噪。
    img_vst_list: [当前帧VST, 前一帧VST]
    offsets_3d: 维度为 (H, W, K_total, 3)，最后一维是 (t, dy, dx)
                其中 t=0 代表当前帧，t=1 代表前一帧
    """
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

    for y in trange(r, H - r, step, desc="BM3D Video Transform"):
        for x in range(r, W - r, step):
            # 记录有效的坐标信息: (t, y, x)
            coords_3d = []  # 必定包含当前帧自身的中心块

            for k in range(K_total):
                t, dy, dx = offsets_3d[y, x, k]
                if t == -1:  # 如果初始化的无效值被传入，跳过
                    continue

                ny, nx = y + dy, x + dx
                if r <= ny <= H - patch_size + r and r <= nx <= W - patch_size + r:
                    coords_3d.append((t, ny, nx))

            K_actual = len(coords_3d)
            if K_actual <= 1:
                continue

            # 从不同帧中提取 Patch 堆叠
            group_3d = np.zeros((K_actual, patch_size, patch_size), dtype=np.float64)
            for i, (ct, cy, cx) in enumerate(coords_3d):
                group_3d[i] = img_vst_list[ct][cy - r: cy - r + patch_size, cx - r: cx - r + patch_size]

            # --- 下面的 3D 变换和滤波代码与你原来完全一样 ---
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

            # ================== 【核心修复：在聚合时乘上 kaiser_2d】 ==================
            for i, (ct, cy, cx) in enumerate(coords_3d):
                if ct == 0:
                    numerator[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size] += group_3d_denoised[
                                                                                               i] * weight * kaiser_2d
                    denominator[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size] += weight * kaiser_2d

    mask = denominator > 0
    denoised_img = curr_img_vst.copy()
    denoised_img[mask] = numerator[mask] / denominator[mask]

    return denoised_img


def process_video_sequence(png_folder, out_folder, K_spatial=7, K_time=3, patch_size=6, step=2):
    # 获取所有图片路径并排序
    png_files = sorted([f for f in os.listdir(png_folder) if f.endswith('.png')])
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    csv_file_path = os.path.join(out_folder, f"AKNN_VBM3D_metrics_K{K_spatial}_{K_time}.csv")

    # 打开文件 (使用 with 语句可以确保后续不出错，但在大循环里我们需要手动控制，这里用普通 open)
    csv_file = open(csv_file_path, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # 写入表头
    csv_writer.writerow(['Frame_Index', 'Filename', 'PSNR(dB)', 'SSIM'])
    print(f"Metrics will be saved to: {csv_file_path}")


    prev_y_vst = None  # 模拟硬件里的前一帧缓存
    time_search_radius = 7  # 硬件时域小窗的半径

    for frame_idx, filename in enumerate(png_files):
        print(f"\n========== Processing Frame {frame_idx}: {filename} ==========")
        img_path = os.path.join(png_folder, filename)

        # 1. 读图并转 YUV (只在 Y 通道做降噪演示)
        y, cb, cr, img_bgr = read_png_to_yuv(img_path)

        # [可选] 添加人工噪声用于测试
        y_noisy = add_poisson_gaussian_noise(y, a=0.02, sigma_norm=25 / 255)

        # 2. VST 变换
        # 注意：由于是普通 PNG，参数 a 和 sigma 需设为经验固定值
        curr_y_vst = forward_gat(y_noisy, a=0.02, sigma=25 / 255)
        H, W = curr_y_vst.shape

        curr_guide_vst = cv2.GaussianBlur(curr_y_vst, (5, 5), 1.5)


        # 3. 空域 AKNN (只在当前帧跑)
        offsets_spatial, dists_spatial = initialize_aknn(curr_guide_vst, K_spatial, patch_size, step)
        offsets_spatial, dists_spatial = run_aknn_pure_python(
            curr_guide_vst, offsets_spatial, dists_spatial, iterations=2,
            patch_size=patch_size, sigma_norm=1.0, step=step
        )

        # 设定硬件 3D 变换的绝对深度
        K_target_power_of_2 = 8
        K_time_max = 3  # 时域选 4 个参战
        K_spatial = 7  # 空域选 7 个参战

        offsets_3d = np.full((H, W, K_target_power_of_2, 3), -1, dtype=int)

        # 硬件监控寄存器
        temporal_usage_stats = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
        total_3d_groups = 0

        for cy in range(patch_size // 2, H - patch_size // 2, step):
            for cx in range(patch_size // 2, W - patch_size // 2, step):

                # 1. 雷打不动：参考块自己 (必定在第0位)
                offsets_3d[cy, cx, 0] = [0, 0, 0]

                # ====== 时空竞争池 (Pool) ======
                candidate_pool = []

                #2. 时域选手进池子 (带着它们的真实距离)
                if prev_y_vst is not None:
                    time_cands = temporal_local_search(
                        curr_guide_vst, prev_y_vst, cy, cx, patch_size, K_time_max, search_radius=4
                    )
                    for dist, dy, dx in time_cands:
                        candidate_pool.append((dist, 1, dy, dx))  # t=1 代表时域

                # 3. 空域选手进池子 (带着它们的真实距离)
                for k_s in range(offsets_spatial.shape[2]):
                    dy, dx = offsets_spatial[cy, cx, k_s, 0], offsets_spatial[cy, cx, k_s, 1]
                    dist = dists_spatial[cy, cx, k_s]  # 直接从 AKNN 结果里取距离

                    if not (dy == 0 and dx == 0):  # 排除掉自己
                        candidate_pool.append((dist, 0, dy, dx))  # t=0 代表空域

                # 4. 全局大排序！只取前 7 名
                candidate_pool.sort(key=lambda item: item[0])
                best_7 = candidate_pool[:7]

                # 5. 把胜利者装入流水线
                valid_count = 1  # 已经有参考块了
                time_used = 0

                for dist, t, dy, dx in best_7:
                    offsets_3d[cy, cx, valid_count] = [t, dy, dx]
                    valid_count += 1
                    if t == 1:
                        time_used += 1

                # 6. 兜底 (理论上只要池子大于7就不会触发)
                while valid_count < K_target_power_of_2:
                    offsets_3d[cy, cx, valid_count] = [0, 0, 0]
                    valid_count += 1

                # 统计
                if time_used in temporal_usage_stats:
                    temporal_usage_stats[time_used] += 1
                total_3d_groups += 1

        # ====== 打印统计信息 ======
        if prev_y_vst is not None:
            print("\n  [硬件监控] 时空竞争结果 (最高时域占比分析):")
            for k in range(8):  # 最多可能7个都是时域的（如果你池子全放开的话，当前限制了4个）
                if k > K_time_max: break
                count = temporal_usage_stats[k]
                pct = (count / total_3d_groups) * 100
                bar = "█" * int(pct / 5)
                print(f"    包含 {k} 个时域块的组数: {count:5d} ({pct:5.1f}%) | {bar}")
            avg_temp = sum(k * v for k, v in temporal_usage_stats.items()) / total_3d_groups
            print(f"    平均每组使用时域块数: {avg_temp:.2f} / 8\n")
        # ====================================

        # 6. 传入跨帧 BM3D 引擎
        img_vst_list = [curr_y_vst, prev_y_vst] if prev_y_vst is not None else [curr_y_vst]
        denoised_y_vst = bm3d_1st_stage_video(img_vst_list, offsets_3d, patch_size, step)

        # 7. 逆 VST 变换
        denoised_y = inverse_gat(denoised_y_vst, a=0.02, sigma=25 / 255)
        denoised_y = np.clip(denoised_y, 0.0, 1.0)
        current_psnr = psnr(y, denoised_y, data_range=1.0)
        current_ssim = ssim(y, denoised_y, data_range=1.0)
        print(f"Final PSNR: {current_psnr:.2f} dB | SSIM: {current_ssim:.4f}")
        csv_writer.writerow([frame_idx, filename, round(current_psnr, 4), round(current_ssim, 4)])
        csv_file.flush()  # 极其重要：写完立刻刷入硬盘，防止程序崩溃丢数据
        # 可视化保存
        showPic(img_bgr, y, y_noisy, cb, cr, denoised_y, out_folder, frame_idx)

        # 8. 更新前一帧缓存 (硬件流水线数据滚动)
        prev_y_vst = None  # 可以拿去噪后的图做下一帧参考，也可以拿原始 VST 图

    csv_file.close()
    print(f"\nAll frames processed. Metrics saved successfully to {csv_file_path}")


if __name__ == "__main__":
    png_folder = "data/Xiph_org_Video/coastguard_60pixel"  # 替换为你的输入文件夹路径
    out_folder = "data/Xiph_org_Video/coastguard_60pixel/res"  # 替换为你的输出文件夹路径
    # png_folder = "data/DAVIS/car-turn"  # 替换为你的输入文件夹路径
    # out_folder = "data/DAVIS/car-turn/res"  # 替换为你的输出文件夹路径
    process_video_sequence(png_folder, out_folder, K_spatial=7, K_time=3, patch_size=7, step=2)