import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import njit, prange
from tqdm import tqdm, trange
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numba import njit, prange
import time
from scipy.fft import dctn, idctn  # 引入 3D 变换库
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from AKNN_init import initialize_aknn,visualize_pixel_and_candidates


def calc_dct_distance(p_ref, p_cand, C, C_T, threshold):
    """计算经过 2D DCT 硬阈值预滤波后的距离"""
    # 1. 2D DCT 变换: F = C * P * C^T
    ref_freq = np.dot(np.dot(C, p_ref), C_T)
    cand_freq = np.dot(np.dot(C, p_cand), C_T)

    # 2. 硬阈值截断：绝对值小于阈值的置为 0 (剔除噪声)
    ref_freq[np.abs(ref_freq) < threshold] = 0.0
    cand_freq[np.abs(cand_freq) < threshold] = 0.0

    # 3. 计算频域差值的平方和 (等价于去噪后的空域距离)
    diff = ref_freq - cand_freq
    return np.sum(diff * diff)

def get_dct_matrix(N):
    """生成 NxN 的一维 DCT-II 变换矩阵"""
    C = np.zeros((N, N), dtype=np.float64)
    for k in range(N):
        for n in range(N):
            if k == 0:
                C[k, n] = 1.0 / np.sqrt(N)
            else:
                C[k, n] = np.sqrt(2.0 / N) * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    return C


# --- 1. 核心辅助函数：维护优先队列 ---
def update_best_k(img, y, x, prop_dy, prop_dx, offsets, dists, patch_size, H, W, K, C, C_T, threshold):
    ny, nx = y + prop_dy, x + prop_dx
    r = patch_size // 2

    # 1. 越界检查 (注意修正了奇偶数兼容的写法)
    if y - r < 0 or y - r + patch_size > H or x - r < 0 or x - r + patch_size > W:
        return
    if ny - r < 0 or ny - r + patch_size > H or nx - r < 0 or nx - r + patch_size > W:
        return

    # 2. 提取 Patch
    patch_src = img[y - r: y - r + patch_size, x - r: x - r + patch_size]
    patch_tgt = img[ny - r: ny - r + patch_size, nx - r: nx - r + patch_size]

    if patch_src.shape != (patch_size, patch_size) or patch_tgt.shape != (patch_size, patch_size):
        return

    # 3. 使用魔改后的 DCT 距离替换普通的 SSD！
    new_dist = calc_dct_distance(patch_src, patch_tgt, C, C_T, threshold)

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


# --- 2. 传播步骤 (Propagation) ---
def propagation_step(img, offsets, dists, patch_size, iter_num, C, C_T, threshold):
    H, W = img.shape[:2]
    K = offsets.shape[2]
    print(f"  > Propagation (Direction: {'Scanline' if iter_num % 2 == 0 else 'Reverse'})...")

    if iter_num % 2 == 0:
        y_range = range(1, H)
        x_range = range(1, W)
        neighbor_deltas = [(-1, 0), (0, -1)]
    else:
        y_range = range(H - 2, -1, -1)
        x_range = range(W - 2, -1, -1)
        neighbor_deltas = [(1, 0), (0, 1)]

    for y in tqdm(y_range, desc=f"    Prop iter{iter_num + 1}", leave=False):
        for x in x_range:
            for dy_n, dx_n in neighbor_deltas:
                nb_y, nb_x = y + dy_n, x + dx_n
                nb_offsets = offsets[nb_y, nb_x]
                for k in range(K):
                    prop_dy, prop_dx = nb_offsets[k]
                    update_best_k(img, y, x, prop_dy, prop_dx,
                                  offsets, dists, patch_size, H, W, K, C, C_T, threshold)


# --- 3. 随机搜索步骤 (Random Search) ---
def random_search_step(img, offsets, dists, patch_size, search_radius, C, C_T, threshold):
    H, W = img.shape[:2]
    K = offsets.shape[2]
    print(f"  > Random Search (Radius: {search_radius:.2f})...")

    for y in tqdm(range(H), desc="    Random", leave=False):
        for x in range(W):
            for k in range(K):
                best_dy, best_dx = offsets[y, x, k]
                rand_dy = int(round(search_radius * np.random.randn()))
                rand_dx = int(round(search_radius * np.random.randn()))
                search_dy = best_dy + rand_dy
                search_dx = best_dx + rand_dx

                update_best_k(img, y, x, search_dy, search_dx,
                              offsets, dists, patch_size, H, W, K, C, C_T, threshold)


# --- 4. 主程序：把所有步骤串起来 ---
def run_aknn_pure_python(img, init_offsets, init_dists, iterations, patch_size, sigma_norm):
    """
    注意：在入口处需要传入 sigma_norm，用来计算 threshold
    """
    H, W = img.shape[:2]
    K = init_offsets.shape[2]
    offsets = init_offsets.copy()
    dists = init_dists.copy()
    search_radius = W

    print(f"Starting AKNN Loop ({iterations} iterations)...")

    # ========== 准备 DCT 矩阵和阈值 ==========
    C_matrix = get_dct_matrix(patch_size)
    C_T_matrix = C_matrix.T

    # 根据 BM3D 论文，2D 预滤波匹配时的硬阈值推荐为 2.5 * sigma 或者更激进一点
    dct_threshold = 2.5 * sigma_norm
    # =========================================

    for i in trange(iterations, desc="AKNN Iter"):
        t0 = time.time()

        # 将矩阵和阈值传下去
        propagation_step(img, offsets, dists, patch_size, i, C_matrix, C_T_matrix, dct_threshold)

        current_radius = search_radius * (0.5 ** i)
        if current_radius < 1: current_radius = 1

        random_search_step(img, offsets, dists, patch_size, current_radius, C_matrix, C_T_matrix, dct_threshold)

        t1 = time.time()
        tqdm.write(f"Iteration {i + 1} finished in {t1 - t0:.2f}s")

    return offsets, dists

    # --- 新增：BM3D 第一阶段 (协同硬阈值滤波) ---
def bm3d_1st_stage(noisy_img, offsets, patch_size, sigma, step=3):
    """
    利用 AKNN 找出的相似块 (offsets) 进行 3D 变换和硬阈值降噪
    step: 滑动步长。设置为 3 或 4 可以成倍加速，同时肉眼几乎看不出质量下降。
    """
    H, W = noisy_img.shape
    K = offsets.shape[2]
    r = patch_size // 2

    # 初始化聚合画布 (分子存放像素累加值，分母存放权重累加值)
    numerator = np.zeros_like(noisy_img, dtype=np.float64)
    denominator = np.zeros_like(noisy_img, dtype=np.float64)

    # 硬阈值 (通常为 2.7 * sigma)
    lambda_3d = 2.7 * sigma

    print(f"Starting BM3D 1st Stage (Hard Thresholding)...")

    # 使用步长 step 遍历图像，极大节省时间
    for y in trange(r, H - r, step, desc="BM3D 3D Transform"):
        for x in range(r, W - r, step):

            # 1. Grouping: 将相似块堆叠成 3D 矩阵
            group_3d = np.zeros((K, patch_size, patch_size), dtype=np.float64)
            valid_k = 0
            actual_coords = []

            for k in range(K):
                dy, dx = offsets[y, x, k]
                ny, nx = y + dy, x + dx

                # 确保提取的块不越界
                if r <= ny < H - r and r <= nx < W - r:
                    group_3d[valid_k] = noisy_img[ny - r:ny + r + 1, nx - r:nx + r + 1]
                    actual_coords.append((ny, nx))
                    valid_k += 1

            if valid_k == 0: continue

            # 截断无效的层
            group_3d = group_3d[:valid_k]

            # 2. 3D Transform (3D DCT)
            # norm='ortho' 保证变换是正交的，能量守恒
            group_3d_freq = dctn(group_3d, norm='ortho')

            # 3. Hard Thresholding (硬阈值过滤)
            # 小于阈值的系数直接置 0 (滤除噪声)
            group_3d_freq[np.abs(group_3d_freq) < lambda_3d] = 0

            # 计算聚合权重: 非零系数个数的倒数 (非零系数越少，说明去噪越彻底，权重越大)
            n_nonzero = np.sum(group_3d_freq != 0)
            weight = 1.0 / n_nonzero if n_nonzero > 0 else 1.0

            # 4. Inverse 3D Transform (逆 3D DCT)
            group_3d_denoised = idctn(group_3d_freq, norm='ortho')

            # 5. Aggregation (加权聚合贴回画布)
            for i in range(valid_k):
                ny, nx = actual_coords[i]
                numerator[ny - r:ny + r + 1, nx - r:nx + r + 1] += group_3d_denoised[i] * weight
                denominator[ny - r:ny + r + 1, nx - r:nx + r + 1] += weight

    # 生成最终去噪图像 (分子除以分母)
    mask = denominator > 0
    denoised_img = noisy_img.copy()
    denoised_img[mask] = numerator[mask] / denominator[mask]

    # 限制范围在 [0, 1]
    return np.clip(denoised_img, 0, 1)

if __name__ == "__main__":

    clean_path = "data/classic_photo/lena_gray_left_up.png"
    clean_img_cv = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
    if clean_path is None:
        print(f"错误：找不到路径为 {clean_path} 的图片，请检查路径。")
    img_clean = clean_img_cv.astype(np.float32) / 255.0

    sigma_val = 25
    sigma_norm = sigma_val / 255.0
    np.random.seed(42)  # 固定种子方便复现
    noise = np.random.normal(0, sigma_norm, img_clean.shape)
    img_noisy = np.clip(img_clean + noise, 0, 1)

    K = 15  # 我们想找 5 个最近邻
    patch_size = 9 # 补丁大小
    offsets, dists = initialize_aknn(img_noisy, K, patch_size)
    final_offsets, final_dists = run_aknn_pure_python(img_noisy, offsets, dists, 4,patch_size,sigma_norm)
    visualize_pixel_and_candidates(
        img_noisy,
        32,
        32,
        final_offsets,
        patch_size
    )
    visualize_pixel_and_candidates(
        img_noisy,
        64,
        64,
        final_offsets,
        patch_size
    )

    img_denoised = bm3d_1st_stage(
        noisy_img=img_noisy,
        offsets=final_offsets,
        patch_size=patch_size,
        sigma=sigma_norm,
        step=4  # 步长为 3 提速
    )
    current_psnr = psnr(img_clean, img_denoised, data_range=1.0)
    current_ssim = ssim(img_clean, img_denoised, data_range=1.0)
    print(f"PSNR: {current_psnr:.2f} dB | SSIM: {current_ssim:.4f}\n")
    # 5. 可视化对比
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1);
    plt.title("Clean (Ground Truth)");
    plt.imshow(img_clean, cmap='gray');
    plt.axis('off')
    plt.subplot(1, 3, 2);
    plt.title(f"Noisy (Sigma={sigma_val})");
    plt.imshow(img_noisy, cmap='gray');
    plt.axis('off')
    plt.subplot(1, 3, 3);
    plt.title("BM3D 1st Stage Denoised");
    plt.imshow(img_denoised, cmap='gray');
    plt.axis('off')
    plt.tight_layout()
    plt.show()