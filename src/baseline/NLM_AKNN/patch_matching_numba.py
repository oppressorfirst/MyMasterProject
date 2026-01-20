import numpy as np
import cv2
import time
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from numba import njit, prange
import math


# =========================
# 0) 可视化（Python，不进 Numba）
# =========================
def visualize_pixel_and_candidates(img, y0, x0, offsets, patch_size, title=None):
    """
    红框：中心 patch
    蓝框：K 个候选 patch
    """
    H, W = img.shape
    K = offsets.shape[2]
    r = patch_size // 2

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.set_title(title if title else f"Pixel ({y0}, {x0}) and its {K} candidates")
    ax.axis("off")

    # 中心 patch（红）
    ax.add_patch(
        patches.Rectangle(
            (x0 - r, y0 - r), patch_size, patch_size,
            linewidth=2, edgecolor="red", facecolor="none"
        )
    )

    # 候选 patch（蓝）
    for k in range(K):
        dy = offsets[y0, x0, k, 0]
        dx = offsets[y0, x0, k, 1]
        ny, nx = y0 + dy, x0 + dx
        if 0 <= ny < H and 0 <= nx < W:
            ax.add_patch(
                patches.Rectangle(
                    (nx - r, ny - r), patch_size, patch_size,
                    linewidth=1.5, edgecolor="blue", facecolor="none"
                )
            )
            ax.text(nx, ny, f"{k}", color="blue", fontsize=10)
    plt.show()


# =========================
# 1) Numba 小工具
# =========================
@njit
def _randn_box_muller():
    """
    Numba 里稳定生成标准正态 N(0,1)：
    Box-Muller: Z = sqrt(-2 ln U1) * cos(2πU2)
    """
    u1 = np.random.random()
    u2 = np.random.random()
    # 防止 log(0)
    if u1 < 1e-12:
        u1 = 1e-12
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


@njit
def _insert_topk(new_dist, new_dy, new_dx, dists_1px, offs_1px, K):
    """
    将 (new_dist, new_dy, new_dx) 插入到某个像素的 top-K（升序）中。
    dists_1px: shape (K,)
    offs_1px : shape (K,2)
    """
    # 不如最差的，直接扔
    if new_dist >= dists_1px[K - 1]:
        return

    # 查重（同偏移不重复加入）
    for k in range(K):
        if offs_1px[k, 0] == new_dy and offs_1px[k, 1] == new_dx:
            return

    # 找插入位置
    pos = -1
    for k in range(K):
        if new_dist < dists_1px[k]:
            pos = k
            break

    if pos == -1:
        return

    # 右移挤掉最差
    for t in range(K - 1, pos, -1):
        dists_1px[t] = dists_1px[t - 1]
        offs_1px[t, 0] = offs_1px[t - 1, 0]
        offs_1px[t, 1] = offs_1px[t - 1, 1]

    dists_1px[pos] = new_dist
    offs_1px[pos, 0] = new_dy
    offs_1px[pos, 1] = new_dx


@njit
def _ssd_patch(img, y, x, ny, nx, patch_size):
    """
    计算两个 patch 的 SSD，假设双方 patch 都合法（不会越界）。
    img: (H,W) float32/float64
    """
    r = patch_size // 2
    s = 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            diff = img[y + dy, x + dx] - img[ny + dy, nx + dx]
            s += diff * diff
    return s


@njit
def _is_patch_valid(y, x, H, W, r):
    return (y - r >= 0) and (y + r < H) and (x - r >= 0) and (x + r < W)


# =========================
# 2) 初始化（Numba，可并行）
# =========================
@njit(parallel=True)
def initialize_aknn_numba(img, K, patch_size=7, seed=123):
    """
    初始化：每个像素生成 K 个高斯随机偏移（sigma_s=W/3），计算距离后填 top-K。
    返回：
      offsets: (H,W,K,2) int32 (dy,dx)
      dists  : (H,W,K)   float32/float64
    """
    np.random.seed(seed)

    H, W = img.shape
    r = patch_size // 2
    sigma_s = W / 3.0

    offsets = np.zeros((H, W, K, 2), dtype=np.int32)
    dists = np.full((H, W, K), np.inf, dtype=np.float32)

    # 每个像素独立：可以并行
    for y in prange(H):
        for x in range(W):
            # 源 patch 不合法：直接跳过（保持 inf）
            if not _is_patch_valid(y, x, H, W, r):
                continue

            d1 = dists[y, x]
            o1 = offsets[y, x]

            for kk in range(K):
                # 生成一个高斯偏移并四舍五入成像素
                dy = int(round(sigma_s * _randn_box_muller()))
                dx = int(round(sigma_s * _randn_box_muller()))
                ny = y + dy
                nx = x + dx

                # 目标 patch 不合法：重采样若干次；仍失败则随机合法点
                tries = 0
                while tries < 8 and (not _is_patch_valid(ny, nx, H, W, r)):
                    dy = int(round(sigma_s * _randn_box_muller()))
                    dx = int(round(sigma_s * _randn_box_muller()))
                    ny = y + dy
                    nx = x + dx
                    tries += 1

                if not _is_patch_valid(ny, nx, H, W, r):
                    # 随机合法点
                    ny = r + int(np.random.random() * (H - 2 * r))
                    nx = r + int(np.random.random() * (W - 2 * r))
                    dy = ny - y
                    dx = nx - x

                dist = _ssd_patch(img, y, x, ny, nx, patch_size)
                _insert_topk(dist, dy, dx, d1, o1, K)

    return offsets, dists


# =========================
# 3) 更新（Numba）
# =========================
@njit
def update_best_k_numba(img, y, x, prop_dy, prop_dx, offsets, dists, patch_size, H, W, K):
    """
    尝试用 (prop_dy,prop_dx) 更新 (y,x) 的 top-K
    """
    r = patch_size // 2

    # 源 patch / 目标 patch 合法性
    if not _is_patch_valid(y, x, H, W, r):
        return

    ny = y + prop_dy
    nx = x + prop_dx
    if not _is_patch_valid(ny, nx, H, W, r):
        return

    dist = _ssd_patch(img, y, x, ny, nx, patch_size)
    _insert_topk(dist, prop_dy, prop_dx, dists[y, x], offsets[y, x], K)


# =========================
# 4) 传播（Numba，加速但保持“顺序依赖”）
#    注意：这里不 parallel=True，避免破坏抄作业依赖
# =========================
@njit
def propagation_step_numba(img, offsets, dists, patch_size, iter_num):
    H, W = img.shape
    K = offsets.shape[2]
    r = patch_size // 2

    if iter_num % 2 == 0:
        # 正向扫描：上、左
        for y in range(r, H - r):
            for x in range(r, W - r):
                # 上邻居
                nb_y, nb_x = y - 1, x
                for k in range(K):
                    dy = offsets[nb_y, nb_x, k, 0]
                    dx = offsets[nb_y, nb_x, k, 1]
                    update_best_k_numba(img, y, x, dy, dx, offsets, dists, patch_size, H, W, K)

                # 左邻居
                nb_y, nb_x = y, x - 1
                for k in range(K):
                    dy = offsets[nb_y, nb_x, k, 0]
                    dx = offsets[nb_y, nb_x, k, 1]
                    update_best_k_numba(img, y, x, dy, dx, offsets, dists, patch_size, H, W, K)
    else:
        # 反向扫描：下、右
        for y in range(H - r - 1, r - 1, -1):
            for x in range(W - r - 1, r - 1, -1):
                # 下邻居
                nb_y, nb_x = y + 1, x
                for k in range(K):
                    dy = offsets[nb_y, nb_x, k, 0]
                    dx = offsets[nb_y, nb_x, k, 1]
                    update_best_k_numba(img, y, x, dy, dx, offsets, dists, patch_size, H, W, K)

                # 右邻居
                nb_y, nb_x = y, x + 1
                for k in range(K):
                    dy = offsets[nb_y, nb_x, k, 0]
                    dx = offsets[nb_y, nb_x, k, 1]
                    update_best_k_numba(img, y, x, dy, dx, offsets, dists, patch_size, H, W, K)


# =========================
# 5) 随机搜索（Numba，可并行）
# =========================
@njit(parallel=True)
def random_search_step_numba(img, offsets, dists, patch_size, radius, seed):
    """
    对每个像素的每个候选，在其附近做随机扰动并尝试更新。
    这一阶段每个像素只更新自己的 top-K，不会互相写，适合 parallel。
    """
    np.random.seed(seed)

    H, W = img.shape
    K = offsets.shape[2]
    r = patch_size // 2

    for y in prange(H):
        for x in range(W):
            if not _is_patch_valid(y, x, H, W, r):
                continue

            for k in range(K):
                best_dy = offsets[y, x, k, 0]
                best_dx = offsets[y, x, k, 1]

                # 正态扰动（半径缩放）
                rd_y = int(round(radius * _randn_box_muller()))
                rd_x = int(round(radius * _randn_box_muller()))

                cand_dy = best_dy + rd_y
                cand_dx = best_dx + rd_x

                update_best_k_numba(img, y, x, cand_dy, cand_dx, offsets, dists, patch_size, H, W, K)


# =========================
# 6) 主循环（Python + tqdm；核心步骤在 Numba 里）
# =========================
def run_aknn_patchmatch_numba(img, K=5, patch_size=7, iterations=4, seed=123):
    """
    返回 final_offsets, final_dists
    """
    # 初始化（Numba）
    offsets, dists = initialize_aknn_numba(img, K, patch_size, seed)

    H, W = img.shape
    base_radius = W / 3.0

    # 主迭代（Python 控制进度条；传播/随机搜索在 Numba）
    for it in trange(iterations, desc="AKNN Iter"):
        t0 = time.time()

        propagation_step_numba(img, offsets, dists, patch_size, it)

        radius = base_radius * (0.5 ** it)
        if radius < 1.0:
            radius = 1.0

        # 为了每次迭代随机不同：seed + it*10007
        random_search_step_numba(img, offsets, dists, patch_size, radius, seed + it * 10007)

        t1 = time.time()
        # tqdm 不乱条
        # （你如果嫌输出多，也可以删掉）
        # print(f"iter {it+1}: {t1-t0:.2f}s")

    return offsets, dists


# =========================
# 7) Demo main（按你项目路径）
# =========================
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "data"
    noisy_path = DATA_DIR / "classic_photo_AWGN_sigma20_seed123456" / "lena_gray_sigma20_seed123456.png"

    noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
    if noisy_img is None:
        raise FileNotFoundError(f"找不到图片：{noisy_path}")

    img_float = (noisy_img.astype(np.float32) / 255.0)

    K = 4
    patch_size = 8
    iterations = 3
    seed = 123

    # 跑 Numba 版
    t0 = time.perf_counter()
    final_offsets, final_dists = run_aknn_patchmatch_numba(
        img_float, K=K, patch_size=patch_size, iterations=iterations, seed=seed
    )
    t1 = time.perf_counter()
    print(f"run_aknn_patchmatch_numba took {t1 - t0:.3f} s")
    print("Done.")
    print("offsets:", final_offsets.shape, final_offsets.dtype)
    print("dists  :", final_dists.shape, final_dists.dtype)

    # 看两个点
    visualize_pixel_and_candidates(img_float, 32, 32, final_offsets, patch_size, title="After PatchMatch (32,32)")
    visualize_pixel_and_candidates(img_float, 64, 64, final_offsets, patch_size, title="After PatchMatch (64,64)")
    visualize_pixel_and_candidates(img_float, 280, 128, final_offsets, patch_size, title="After PatchMatch (64,64)")

