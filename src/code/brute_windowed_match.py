from pathlib import Path
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim



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


# =========================
# Step 1 - CONFIG
# =========================
clean_path = "data/classic_photo/lena_gray.png"
clean_img_cv = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
if clean_path is None:
    print(f"错误：找不到路径为 {clean_path} 的图片，请检查路径。")
img_clean = clean_img_cv.astype(np.float32) / 255.0
sigma_val = 25
sigma_norm = sigma_val / 255.0
np.random.seed(42)  # 固定种子方便复现
# noise = np.random.normal(0, sigma_norm, img_clean.shape)
noisy_img_float = add_poisson_gaussian_noise(img_clean, a=0.02,sigma_norm=sigma_norm, seed=42)
# noisy_img_float = np.clip(img_clean + noise, 0, 1)


patch_size = 8
top_k = 8
window_size = 39

stride = patch_size // 2  # 不重叠
search_radius = window_size // 2

# =========================
# Step 2 - Load
# =========================

H, W = noisy_img_float.shape

# =========================
# Step 3 - Brute-force windowed matching + timing
# =========================
from tqdm import tqdm
import time

guide_img = cv2.GaussianBlur(noisy_img_float, (5, 5), 1.5)

# =========================
# Step 3 - Brute-force windowed matching + timing (with tqdm)
# =========================
all_results = []

# 预生成 ref 坐标列表，tqdm 才能准确显示 total
ref_ys = list(range(0, H - patch_size + 1, stride))
ref_xs = list(range(0, W - patch_size + 1, stride))
total = len(ref_ys) * len(ref_xs)

t_global0 = time.perf_counter()

pbar = tqdm(total=total, desc="Brute-force matching", dynamic_ncols=True)
C_matrix = get_dct_matrix(patch_size)
C_T_matrix = C_matrix.T

for ref_y in ref_ys:
    for ref_x in ref_xs:
        t0 = time.perf_counter()

        ref_patch = guide_img[
            ref_y:ref_y + patch_size,
            ref_x:ref_x + patch_size
        ]

        matches = []

        # ---------- ① 限制搜索窗口 ----------
        y_min = max(0, ref_y - search_radius)
        y_max = min(H - patch_size, ref_y + search_radius)

        x_min = max(0, ref_x - search_radius)
        x_max = min(W - patch_size, ref_x + search_radius)

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                # ---------- ② 排除 self ----------
                if y == ref_y and x == ref_x:
                    continue

                cand_patch = guide_img[y:y + patch_size, x:x + patch_size]

                # dist = calc_dct_distance(ref_patch, cand_patch, C_matrix, C_T_matrix, 2.5 * sigma_norm)
                diff = ref_patch - cand_patch
                dist = np.mean(diff * diff)

                matches.append((dist, y, x))

        matches.sort(key=lambda t: t[0])
        top_matches = matches[:top_k]

        t1 = time.perf_counter()

        all_results.append({
            "ref_pos": (ref_y, ref_x),
            "top_matches": top_matches,
            "time": t1 - t0
        })

        # tqdm 更新
        pbar.update(1)

        # 每隔一段更新一次 postfix（避免频繁 set_postfix 带来开销）
        if (pbar.n % 200) == 0 or pbar.n == total:
            elapsed = time.perf_counter() - t_global0
            avg_time = elapsed / pbar.n
            best_dist = top_matches[0][0] if len(top_matches) > 0 else float("nan")
            pbar.set_postfix({
                "elapsed_s": f"{elapsed:.1f}",
                "avg_s/ref": f"{avg_time:.4f}",
                "best": f"{best_dist:.2e}",
            })

pbar.close()

t_global1 = time.perf_counter()

print("=" * 60)
print(f"Total brute-force windowed matching time: {t_global1 - t_global0:.3f} s")
print("=" * 60)


# =========================
# Step 4 - 可视化某一个 ref
# =========================
idx = 515  # 你原来选的
res = all_results[idx]

ref_y, ref_x = res["ref_pos"]
top_matches = res["top_matches"]

vis = (noisy_img_float * 255).astype(np.uint8)
vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

# ref patch（绿）
cv2.rectangle(
    vis,
    (ref_x, ref_y),
    (ref_x + patch_size - 1, ref_y + patch_size - 1),
    (0, 255, 0),
    2
)

# top-k（红）
for _, y, x in top_matches:
    cv2.rectangle(
        vis,
        (x, y),
        (x + patch_size - 1, y + patch_size - 1),
        (0, 0, 255),
        1
    )

# ---------- 显示 ----------
plt.figure(figsize=(6, 6))
plt.title(f"Ref {idx}  (green=ref, red=topK)")
plt.imshow(vis[..., ::-1])
plt.axis("off")
plt.show()


def bm3d_1st_stage_poisson_gaussian(img, all_results, patch_size, a, sigma_norm):
    """
    针对泊松-高斯噪声改进的 BM3D 第一阶段 (自适应局部硬阈值)
    a: 泊松增益 (Photon gain)
    b: 高斯读取噪声方差 (Gaussian variance)
    """
    H, W = img.shape
    numerator = np.zeros_like(img, dtype=np.float64)
    denominator = np.zeros_like(img, dtype=np.float64)

    print("\nStarting Adaptive BM3D 1st Stage Collaborative Filtering...")

    for res in tqdm(all_results, desc="3D Filtering (Adaptive)"):
        ref_y, ref_x = res["ref_pos"]
        matches = res["top_matches"]

        # 1. 整理坐标与堆叠 3D 张量
        coords = [(ref_y, ref_x)] + [(y, x) for dist, y, x in matches]
        K_actual = len(coords)
        group_3d = np.zeros((K_actual, patch_size, patch_size), dtype=np.float64)
        for i, (y, x) in enumerate(coords):
            group_3d[i] = img[y:y + patch_size, x:x + patch_size]

        # ========================================================
        # 【核心改进 1：估计局部亮度与局部噪声标准差】
        # ========================================================
        # 使用参考块 (第 0 层) 的均值作为当前这组相似块的局部真实亮度估计
        local_mean = np.mean(group_3d[0])
        local_mean = max(local_mean, 0.0)  # 防止极端情况出现负值

        # 根据泊松-高斯模型计算局部标准差
        local_sigma2 = a * local_mean + sigma_norm
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
        # 在原版 BM3D 中，权重其实是 1 / (N_nonzero * sigma^2)
        # 以前 sigma 是全局常数所以省略了，现在 sigma 是局部的，必须加进来！
        # 否则亮部块（噪声方差大）的错误累加会污染全局
        n_nonzero = np.sum(group_3d_freq != 0)
        if n_nonzero > 0:
            weight = 1.0 / (n_nonzero * local_sigma2)
        else:
            weight = 1.0 / local_sigma2
        # ========================================================

        # 6. 逆 3D 变换
        group_3d_denoised = idctn(group_3d_freq, norm='ortho')

        # 7. 聚合 (把去噪后的块加权贴回原图)
        for i, (y, x) in enumerate(coords):
            numerator[y:y + patch_size, x:x + patch_size] += group_3d_denoised[i] * weight
            denominator[y:y + patch_size, x:x + patch_size] += weight

    # 8. 归一化输出
    mask = denominator > 0
    denoised_img = img.copy()
    denoised_img[mask] = numerator[mask] / denominator[mask]

    return np.clip(denoised_img, 0, 1)

def bm3d_1st_stage_from_results(img, all_results, patch_size, sigma):
    """
    利用预先计算好的 all_results (匹配坐标) 进行 3D 协同硬阈值降噪
    """
    H, W = img.shape
    numerator = np.zeros_like(img, dtype=np.float64)
    denominator = np.zeros_like(img, dtype=np.float64)

    # 硬阈值 (理论推荐值通常在 2.7 * sigma 左右)
    lambda_3d = 2.7 * sigma

    print("\nStarting BM3D 1st Stage Collaborative Filtering...")

    for res in tqdm(all_results, desc="3D Filtering"):
        ref_y, ref_x = res["ref_pos"]
        matches = res["top_matches"]

        # 1. 整理坐标：【关键】把参考块自身(ref_y, ref_x)放在第一个！
        coords = [(ref_y, ref_x)] + [(y, x) for dist, y, x in matches]
        K_actual = len(coords)

        # 2. 堆叠成 3D 张量
        group_3d = np.zeros((K_actual, patch_size, patch_size), dtype=np.float64)
        for i, (y, x) in enumerate(coords):
            group_3d[i] = img[y:y + patch_size, x:x + patch_size]

        # 3. 3D 变换 (使用 3D DCT)
        group_3d_freq = dctn(group_3d, norm='ortho')

        # 4. 硬阈值截断 (核心降噪步骤)
        group_3d_freq[np.abs(group_3d_freq) < lambda_3d] = 0

        # 5. 计算聚合权重 (非零系数个数的倒数)
        n_nonzero = np.sum(group_3d_freq != 0)
        weight = 1.0 / n_nonzero if n_nonzero > 0 else 1.0

        # 6. 逆 3D 变换
        group_3d_denoised = idctn(group_3d_freq, norm='ortho')

        # 7. 聚合 (把去噪后的块加权贴回原图)
        for i, (y, x) in enumerate(coords):
            numerator[y:y + patch_size, x:x + patch_size] += group_3d_denoised[i] * weight
            denominator[y:y + patch_size, x:x + patch_size] += weight

    # 8. 归一化输出
    mask = denominator > 0
    denoised_img = img.copy()
    denoised_img[mask] = numerator[mask] / denominator[mask]

    return np.clip(denoised_img, 0, 1)


# =========================
# Step 6 - 运行并可视化降噪结果
# =========================

# 假设你的噪声强度 sigma 是 25 (如果输入图本身没加噪，去噪效果可能就是把它变模糊)
sigma_assumed = 25 / 255.0

t_filter0 = time.perf_counter()
# 传入原图、匹配结果列表、patch_size 和 sigma
# denoised_result = bm3d_1st_stage_from_results(noisy_img_float, all_results, patch_size, sigma_assumed)
denoised_result = bm3d_1st_stage_poisson_gaussian(noisy_img_float,all_results, patch_size, a=0.02, sigma_norm=sigma_assumed)
t_filter1 = time.perf_counter()

current_psnr = psnr(img_clean, denoised_result, data_range=1.0)
current_ssim = ssim(img_clean, denoised_result, data_range=1.0)

print(f"PSNR: {current_psnr:.2f} dB | SSIM: {current_ssim:.4f}\n")
print("=" * 60)
print(f"Total 3D Filtering time: {t_filter1 - t_filter0:.3f} s")
print("=" * 60)

# 显示整体结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Noisy Input")
plt.imshow(noisy_img_float, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("BM3D 1st Stage Denoised")
plt.imshow(denoised_result, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

