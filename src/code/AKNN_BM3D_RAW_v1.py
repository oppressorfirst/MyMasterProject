import pathlib
import h5py
import scipy.io as sio
import pandas as pd
from tqdm import tqdm, trange
import cv2
import numpy as np
import time
from scipy.fft import dctn, idctn  # 引入 3D 变换库
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pywt


# --- 4. 主程序：把所有步骤串起来 ---
def run_aknn_pure_python(img, init_offsets, init_dists, iterations, patch_size, sigma_norm, step=1):
    H, W = img.shape[:2]
    offsets = init_offsets.copy()
    dists = init_dists.copy()
    search_radius = W

    print(f"Starting AKNN Loop ({iterations} iterations, Step={step})...")

    for i in trange(iterations, desc="AKNN Iter"):
        t0 = time.time()

        propagation_step(img, offsets, dists, patch_size, i, step)

        current_radius = search_radius * (0.5 ** i)
        if current_radius < 1: current_radius = 1

        random_search_step(img, offsets, dists, patch_size, current_radius, step)

        t1 = time.time()
        tqdm.write(f"Iteration {i + 1} finished in {t1 - t0:.2f}s")

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
    引入 2D Kaiser 窗消除 Patch 拼接伪影。
    """
    H, W = img_vst.shape
    K_offsets = offsets.shape[2]
    r = patch_size // 2

    numerator = np.zeros_like(img_vst, dtype=np.float64)
    denominator = np.zeros_like(img_vst, dtype=np.float64)

    print("\nStarting BM3D 1st Stage on VST domain...")

    # 【核心改进 1】：生成 2D Kaiser 窗
    # beta=2.0 是 BM3D 常用的经验值，中心高，边缘平滑下降
    kaiser_1d = np.kaiser(patch_size, 2.0)
    kaiser_2d = np.outer(kaiser_1d, kaiser_1d)

    # VST 域的全局噪声标准差被稳定为 1.0
    sigma_vst = 1.0
    lambda_3d = 2.7 * sigma_vst
    sigma_vst2 = sigma_vst ** 2

    for y in trange(r, H - r, step, desc="BM3D 3D Transform"):
        for x in range(r, W - r, step):
            coords = [(y, x)] # 包含自己

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

            # 4. 全局硬阈值截断
            n_nonzero = 0
            for i in range(len(haar_coeffs)):
                haar_coeffs[i][np.abs(haar_coeffs[i]) < lambda_3d] = 0
                n_nonzero += np.sum(haar_coeffs[i] != 0)

            # 5. 计算聚合权重
            if n_nonzero > 0:
                weight = 1.0 / (n_nonzero * sigma_vst2)
            else:
                weight = 1.0 / sigma_vst2

            # 6. 逆向混合变换
            group_1d_inv = pywt.waverec(haar_coeffs, 'haar', mode='symmetric', axis=0)
            group_1d_inv = group_1d_inv[:K_actual, :, :]
            group_3d_denoised = idctn(group_1d_inv, axes=(1, 2), norm='ortho')

            # 7. 聚合 【核心改进 2：乘上 Kaiser 窗】
            for i, (cy, cx) in enumerate(coords):
                # 将原来的单纯相加，改为乘以 kaiser_2d 权重
                numerator[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size] += group_3d_denoised[i] * weight * kaiser_2d
                denominator[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size] += weight * kaiser_2d

    mask = denominator > 0
    denoised_img = img_vst.copy()
    denoised_img[mask] = numerator[mask] / denominator[mask]

    return denoised_img

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
    h, w = img.shape[:2]
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
    for y in tqdm(range(r, H - r, step), desc="Init AKNN"):
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

    if patch_src.shape[:2] != (patch_size, patch_size) or patch_tgt.shape[:2] != (patch_size, patch_size):
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

    for y in tqdm(y_range, desc=f"    Prop iter{iter_num + 1}", leave=False):
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
    for y in tqdm(range(r, H - r, step), desc="    Random", leave=False):
        for x in range(r, W - r, step):
            for k in range(K):
                best_dy, best_dx = offsets[y, x, k]
                rand_dy = int(round(search_radius * np.random.randn()))
                rand_dx = int(round(search_radius * np.random.randn()))
                search_dy = best_dy + rand_dy
                search_dx = best_dx + rand_dx

                update_best_k(img, y, x, search_dy, search_dx,
                              offsets, dists, patch_size, H, W, K)






def pack_bayer(raw_img):
    """
    将 (H, W) 的 RAW 图像无视具体颜色，按空间位置打包为 (H/2, W/2, 4)
    通道顺序始终为: 0:(偶行,偶列), 1:(偶行,奇列), 2:(奇行,偶列), 3:(奇行,奇列)
    """
    H, W = raw_img.shape
    packed = np.zeros((H // 2, W // 2, 4), dtype=raw_img.dtype)
    packed[:, :, 0] = raw_img[0::2, 0::2]
    packed[:, :, 1] = raw_img[0::2, 1::2]
    packed[:, :, 2] = raw_img[1::2, 0::2]
    packed[:, :, 3] = raw_img[1::2, 1::2]
    return packed

def unpack_bayer(packed):
    """
    将 (H/2, W/2, 4) 还原为 (H, W) 的 RAW 图像
    """
    H2, W2, C = packed.shape
    raw_img = np.zeros((H2 * 2, W2 * 2), dtype=packed.dtype)
    raw_img[0::2, 0::2] = packed[:, :, 0]
    raw_img[0::2, 1::2] = packed[:, :, 1]
    raw_img[1::2, 0::2] = packed[:, :, 2]
    raw_img[1::2, 1::2] = packed[:, :, 3]
    return raw_img

def get_noise_and_bayer_info(scene_id, noise_csv_path, bayer_csv_path):
    """
    根据 scene_id 提取四个通道的 a(beta1) 和 sigma(sqrt(beta2))
    """
    # 1. 提取相机 ID (例如 '0001_001_S6_...' 提取 'S6')
    camera_id = scene_id.split('_')[2]

    # 2. 获取该相机的 Bayer 排列
    df_bayer = pd.read_csv(bayer_csv_path)
    bayer_pattern = df_bayer[df_bayer['camera_id'] == camera_id]['bayer_pattern'].values[0].lower()

    # 映射四通道(空间位置)对应的颜色 ('r', 'g', 'b')
    # index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    color_map = {
        'rggb': ['r', 'g', 'g', 'b'],
        'bggr': ['b', 'g', 'g', 'r'],
        'grbg': ['g', 'r', 'b', 'g'],
        'gbrg': ['g', 'b', 'r', 'g']
    }
    channel_colors = color_map[bayer_pattern]

    # 3. 获取该场景的噪声参数 beta1, beta2
    df_noise = pd.read_csv(noise_csv_path)
    scene_row = df_noise[df_noise['scene_instance_id'] == scene_id].iloc[0]

    params = []
    for color in channel_colors:
        beta1 = scene_row[f'beta1_{color}']
        beta2 = scene_row[f'beta2_{color}']

        # 针对部分极端拟合情况 beta2 可能为负，截断为0
        a_est = beta1
        sigma_est = np.sqrt(max(beta2, 0.0))
        params.append((a_est, sigma_est))

    return params, bayer_pattern


def process_sidd_raw(gt_path, noisy_path, scene_id, noise_csv, bayer_csv, out_path, K, patch_size, step):
    print(f"Loading data for scene: {scene_id} ...")

    # 1. 读取 MAT 文件
    with h5py.File(gt_path, 'r') as f:
        gt_raw = np.array(f['x']).T.astype(np.float32)
    with h5py.File(noisy_path, 'r') as f:
        noisy_raw = np.array(f['x']).T.astype(np.float32)

    max_val = max(np.max(gt_raw), 1.0)
    if max_val > 10.0:
        gt_raw = gt_raw / max_val
        noisy_raw = noisy_raw / max_val

    # 2. 获取参数并打包 Bayer
    channel_params, bayer_pattern = get_noise_and_bayer_info(scene_id, noise_csv, bayer_csv)
    print(f"Bayer Pattern: {bayer_pattern.upper()}")

    gt_packed = pack_bayer(gt_raw)
    noisy_packed = pack_bayer(noisy_raw)

    vst_packed = np.zeros_like(noisy_packed)

    # 3. 对 4 个通道分别执行前向 VST
    for ch in range(4):
        a_est, sigma_est = channel_params[ch]
        vst_packed[:, :, ch] = forward_gat(noisy_packed[:, :, ch], a_est, sigma_est)

    # ========================================================
    # 【核心改进：生成算距离专用的高斯向导图】
    # ========================================================
    print("\n--- Generating Gaussian Guide Image for AKNN ---")
    guide_packed = np.zeros_like(vst_packed)
    for ch in range(4):
        # 用 3x3 高斯模糊粗略压制高频噪声，避免 AKNN 匹配到长得像的纯噪声块
        guide_packed[:, :, ch] = cv2.GaussianBlur(vst_packed[:, :, ch], (3, 3), 0.5)

    print("\n--- Running Joint AKNN on Macro-Pixels (4 Channels) ---")

    # 【注意：这里传入的是 guide_packed】
    offsets, dists = initialize_aknn(guide_packed, K, patch_size, step=step)
    final_offsets, final_dists = run_aknn_pure_python(
        guide_packed, offsets, dists, iterations=2, patch_size=patch_size, sigma_norm=1.0, step=step
    )

    denoised_packed = np.zeros_like(noisy_packed)

    # 5. 使用联合匹配的 Offsets 进行逐通道的 BM3D 降噪
    for ch in range(4):
        a_est, sigma_est = channel_params[ch]
        print(f"\n--- Running BM3D for Channel {ch + 1}/4 ---")

        # 【极其重要：真正进行 3D 降噪时，必须传入原本带有噪声的 vst_packed，不能用模糊的向导图！】
        vst_noisy_ch = vst_packed[:, :, ch]

        denoised_vst_ch = bm3d_1st_stage_vst_offsets(
            img_vst=vst_noisy_ch,
            offsets=final_offsets,
            patch_size=patch_size,
            step=step
        )

        # 逆向 VST
        denoised_ch = inverse_gat(denoised_vst_ch, a_est, sigma_est)
        denoised_packed[:, :, ch] = np.clip(denoised_ch, 0.0, 1.0)

    # 6. 解包并计算指标
    denoised_raw = unpack_bayer(denoised_packed)

    psnr_val = psnr(gt_raw, denoised_raw, data_range=1.0)
    ssim_val = ssim(gt_raw, denoised_raw, data_range=1.0)

    print("\n================ Results ================")
    print(f"PSNR: {psnr_val:.4f} dB")
    print(f"SSIM: {ssim_val:.4f}")

    sio.savemat(out_path, {'x': denoised_raw})  # 为了兼顾你的 MATLAB demo，这里改成了 'x'
    print(f"Denoised RAW saved to {out_path}")

    return denoised_raw, psnr_val, ssim_val







if __name__ == "__main__":
    # 根据你提供的文件名，通常你需要从包含完整信息的 txt 或者路径名中拿到 scene_id。
    # 这里用你 CSV 中的一行数据为例：

    dataset_path = "data/SIDD_small_RAW/"
    SCENE_NAME = "0065_003_GP_10000_08460_4400_N"
    scene_dir = pathlib.Path(dataset_path) / SCENE_NAME

    NOISE_CSV = pathlib.Path(dataset_path)/"noise_level_functions.csv"
    BAYER_CSV = pathlib.Path(dataset_path)/"bayer_patterns.csv"

    GT_MAT = scene_dir/"GT_RAW_010.mat"  # 你的 GT 路径
    NOISY_MAT = scene_dir/"NOISY_RAW_010.mat"  # 你的 Noisy 路径

    OUT_MAT = scene_dir/"DENOISED_RAW_v1.mat"

    K = 7
    patch_size = 6 # 3,4
    step  = 2 # 或者为4
    process_sidd_raw(GT_MAT, NOISY_MAT, SCENE_NAME, NOISE_CSV, BAYER_CSV, OUT_MAT, K, patch_size, step)