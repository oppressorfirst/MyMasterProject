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


def bm3d_1st_stage_poisson_gaussian_offsets(img, offsets, patch_size, a, sigma_norm, step=3):
    """
    接收 AKNN offsets 的泊松-高斯自适应 BM3D 降噪
    """
    H, W = img.shape
    K_offsets = offsets.shape[2]
    r = patch_size // 2

    numerator = np.zeros_like(img, dtype=np.float64)
    denominator = np.zeros_like(img, dtype=np.float64)

    print("\nStarting Adaptive BM3D 1st Stage with Offsets...")

    # 使用步长 step 遍历图像，极大节省 3D 变换的计算量
    for y in trange(r, H - r, step, desc="BM3D 3D Transform"):
        for x in range(r, W - r, step):

            # 1. 整理坐标：【关键】把参考块自身 (y, x) 强制放在第 0 层！
            coords = [(y, x)]

            # 遍历 offsets 提取这一个像素的 K 个相似块坐标
            for k in range(K_offsets):
                dy, dx = offsets[y, x, k]
                ny, nx = y + dy, x + dx

                # 越界检查 (只保留在图像内部的块)
                # 使用通用的边界计算，兼容奇偶数 patch_size
                if r <= ny <= H - patch_size + r and r <= nx <= W - patch_size + r:
                    coords.append((ny, nx))

            K_actual = len(coords)
            if K_actual <= 1:
                continue  # 如果除了自己以外没找到合法的，就跳过

            # 2. 堆叠成 3D 张量
            group_3d = np.zeros((K_actual, patch_size, patch_size), dtype=np.float64)
            for i, (cy, cx) in enumerate(coords):
                group_3d[i] = img[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size]

            # ========================================================
            # 【核心改进 1：估计局部亮度与局部噪声标准差】
            # ========================================================
            local_mean = np.mean(group_3d[0])
            local_mean = max(local_mean, 0.0)

            # 【已修复！】高斯噪声的方差必须是 sigma_norm 的平方
            local_sigma2 = a * local_mean + (sigma_norm ** 2)
            local_sigma = np.sqrt(max(local_sigma2, 1e-10))

            # 动态计算当前 3D 块的硬阈值
            lambda_3d_local = 2.7 * local_sigma
            # ========================================================

            # 3. 3D 变换 (使用 3D DCT)
            group_3d_freq = dctn(group_3d, norm='ortho')

            # 4. 自适应硬阈值截断
            group_3d_freq[np.abs(group_3d_freq) < lambda_3d_local] = 0

            # ========================================================
            # 【核心改进 2：自适应聚合权重】
            # ========================================================
            n_nonzero = np.sum(group_3d_freq != 0)
            if n_nonzero > 0:
                weight = 1.0 / (n_nonzero * local_sigma2)
            else:
                weight = 1.0 / local_sigma2
            # ========================================================

            # 6. 逆 3D 变换
            group_3d_denoised = idctn(group_3d_freq, norm='ortho')

            # 7. 聚合 (把去噪后的块加权贴回原图)
            for i, (cy, cx) in enumerate(coords):
                numerator[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size] += group_3d_denoised[i] * weight
                denominator[cy - r: cy - r + patch_size, cx - r: cx - r + patch_size] += weight

    # 8. 归一化输出
    mask = denominator > 0
    denoised_img = img.copy()
    denoised_img[mask] = numerator[mask] / denominator[mask]

    return np.clip(denoised_img, 0, 1)



if __name__ == "__main__":

    clean_path = "data/classic_photo/lena_gray.png"
    clean_img_cv = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
    if clean_path is None:
        print(f"错误：找不到路径为 {clean_path} 的图片，请检查路径。")
    img_clean = clean_img_cv.astype(np.float32) / 255.0

    sigma_val = 25
    sigma_norm = sigma_val / 255.0
    np.random.seed(42)  # 固定种子方便复现
    img_noisy = add_poisson_gaussian_noise(img_clean, a=0.02, sigma_norm=sigma_norm, seed=42)
    #noise = np.random.normal(0, sigma_norm, img_clean.shape)
    #img_noisy = np.clip(img_clean + noise, 0, 1)

    guide_img = cv2.GaussianBlur(img_noisy, (5, 5), 1.5)
    K = 7  # 我们想找 5 个最近邻
    patch_size = 7 # 补丁大小
    process_step = 2


    offsets, dists = initialize_aknn(guide_img, K, patch_size, step=process_step)

    final_offsets, final_dists = run_aknn_pure_python(
        guide_img, offsets, dists, 2, patch_size, sigma_norm, step=process_step
    )
    visualize_block_by_index(
        img_noisy,
        block_idx=244,
        offsets=final_offsets,
        patch_size=patch_size,
        step=process_step
    )

    # 你还可以随意试几个别的块，比如第 10 个和第 500 个
    visualize_block_by_index(img_noisy, 10, final_offsets, patch_size, process_step)
    visualize_block_by_index(img_noisy, 500, final_offsets, patch_size, process_step)

    # img_denoised = bm3d_1st_stage(
    #     noisy_img=img_noisy,
    #     offsets=final_offsets,
    #     patch_size=patch_size,
    #     sigma=sigma_norm,
    #     step=1  # 步长为 3 提速
    # )

    img_denoised = bm3d_1st_stage_poisson_gaussian_offsets(
        img=img_noisy,
        offsets=final_offsets,
        patch_size=patch_size,
        a=0.03,
        sigma_norm=sigma_norm,
        step=process_step  # 这里与上面的 step=4 保持一致
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