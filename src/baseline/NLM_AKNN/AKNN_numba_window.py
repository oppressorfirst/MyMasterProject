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

    ax.add_patch(
        patches.Rectangle(
            (x0 - r, y0 - r), patch_size, patch_size,
            linewidth=2, edgecolor="red", facecolor="none"
        )
    )

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
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 < 1e-12:
        u1 = 1e-12
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


@njit
def _is_patch_valid(y, x, H, W, r):
    return (y - r >= 0) and (y + r < H) and (x - r >= 0) and (x + r < W)


@njit
def _is_in_window(y, x, ny, nx, win_radius):
    # 固定正方形窗口约束
    return (abs(ny - y) <= win_radius) and (abs(nx - x) <= win_radius)


@njit
def _ssd_patch(img, y, x, ny, nx, patch_size):
    r = patch_size // 2
    s = 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            diff = img[y + dy, x + dx] - img[ny + dy, nx + dx]
            s += diff * diff
    return s


@njit
def _insert_topk(new_dist, new_dy, new_dx, dists_1px, offs_1px, K):
    if new_dist >= dists_1px[K - 1]:
        return

    # 查重
    for k in range(K):
        if offs_1px[k, 0] == new_dy and offs_1px[k, 1] == new_dx:
            return

    pos = -1
    for k in range(K):
        if new_dist < dists_1px[k]:
            pos = k
            break
    if pos == -1:
        return

    for t in range(K - 1, pos, -1):
        dists_1px[t] = dists_1px[t - 1]
        offs_1px[t, 0] = offs_1px[t - 1, 0]
        offs_1px[t, 1] = offs_1px[t - 1, 1]

    dists_1px[pos] = new_dist
    offs_1px[pos, 0] = new_dy
    offs_1px[pos, 1] = new_dx


# =========================
# 2) 初始化：窗口内随机采样（Numba，可并行）
# =========================
@njit(parallel=True)
def initialize_aknn_numba_window(img, K, patch_size, win_radius, seed):
    """
    初始化：每个像素在自己的窗口内随机取 K 个候选（均匀），计算 SSD 填 top-K
    """
    np.random.seed(seed)

    H, W = img.shape
    r = patch_size // 2

    offsets = np.zeros((H, W, K, 2), dtype=np.int32)
    dists = np.full((H, W, K), np.inf, dtype=np.float32)

    for y in prange(H):
        for x in range(W):
            if not _is_patch_valid(y, x, H, W, r):
                continue

            d1 = dists[y, x]
            o1 = offsets[y, x]

            # 在窗口内抽样 K 次（为了更稳，允许多尝试几次找合法 patch）
            filled = 0
            tries = 0
            while filled < K and tries < K * 20:
                tries += 1

                # 窗口内均匀随机选点
                # ny = y + uniform[-win_radius, win_radius]
                # nx = x + uniform[-win_radius, win_radius]
                dy = int(np.floor((np.random.random() * (2 * win_radius + 1)) - win_radius))
                dx = int(np.floor((np.random.random() * (2 * win_radius + 1)) - win_radius))
                ny = y + dy
                nx = x + dx

                # 必须在窗口内（理论上 dy/dx 已保证），再保证 patch 合法
                if not _is_in_window(y, x, ny, nx, win_radius):
                    continue
                if not _is_patch_valid(ny, nx, H, W, r):
                    continue

                dist = _ssd_patch(img, y, x, ny, nx, patch_size)
                _insert_topk(dist, dy, dx, d1, o1, K)
                filled += 1

            # 如果边缘处导致难以填满（比如窗口内合法 patch 太少）
            # 不强求填满：保持剩余为 inf（后续传播/随机搜索会逐渐填）


    return offsets, dists


# =========================
# 3) 更新：加入窗口约束（Numba）
# =========================
@njit
def update_best_k_numba_window(img, y, x, prop_dy, prop_dx,
                               offsets, dists, patch_size, win_radius, H, W, K):
    r = patch_size // 2

    if not _is_patch_valid(y, x, H, W, r):
        return

    ny = y + prop_dy
    nx = x + prop_dx

    # 关键：窗口约束
    if not _is_in_window(y, x, ny, nx, win_radius):
        return

    if not _is_patch_valid(ny, nx, H, W, r):
        return

    dist = _ssd_patch(img, y, x, ny, nx, patch_size)
    _insert_topk(dist, prop_dy, prop_dx, dists[y, x], offsets[y, x], K)


# =========================
# 4) 传播：保持顺序依赖（Numba，不 parallel）
# =========================
@njit
def propagation_step_numba_window(img, offsets, dists, patch_size, win_radius, iter_num):
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
                    update_best_k_numba_window(
                        img, y, x, dy, dx, offsets, dists, patch_size, win_radius, H, W, K
                    )

                # 左邻居
                nb_y, nb_x = y, x - 1
                for k in range(K):
                    dy = offsets[nb_y, nb_x, k, 0]
                    dx = offsets[nb_y, nb_x, k, 1]
                    update_best_k_numba_window(
                        img, y, x, dy, dx, offsets, dists, patch_size, win_radius, H, W, K
                    )
    else:
        # 反向扫描：下、右
        for y in range(H - r - 1, r - 1, -1):
            for x in range(W - r - 1, r - 1, -1):
                # 下邻居
                nb_y, nb_x = y + 1, x
                for k in range(K):
                    dy = offsets[nb_y, nb_x, k, 0]
                    dx = offsets[nb_y, nb_x, k, 1]
                    update_best_k_numba_window(
                        img, y, x, dy, dx, offsets, dists, patch_size, win_radius, H, W, K
                    )

                # 右邻居
                nb_y, nb_x = y, x + 1
                for k in range(K):
                    dy = offsets[nb_y, nb_x, k, 0]
                    dx = offsets[nb_y, nb_x, k, 1]
                    update_best_k_numba_window(
                        img, y, x, dy, dx, offsets, dists, patch_size, win_radius, H, W, K
                    )


# =========================
# 5) 随机搜索：窗口内抖动（Numba，可并行）
# =========================
@njit(parallel=True)
def random_search_step_numba_window(img, offsets, dists, patch_size, win_radius, radius, seed):
    """
    radius: 随机扰动的尺度（会逐迭代衰减），但最终候选必须落在窗口内
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

                # 正态扰动
                rd_y = int(round(radius * _randn_box_muller()))
                rd_x = int(round(radius * _randn_box_muller()))

                cand_dy = best_dy + rd_y
                cand_dx = best_dx + rd_x

                ny = y + cand_dy
                nx = x + cand_dx

                # 关键：必须仍在窗口内
                if not _is_in_window(y, x, ny, nx, win_radius):
                    continue

                update_best_k_numba_window(
                    img, y, x, cand_dy, cand_dx, offsets, dists, patch_size, win_radius, H, W, K
                )


# =========================
# 6) 主循环（Python + tqdm）
# =========================
def run_aknn_patchmatch_numba_window(img, K=5, patch_size=7, iterations=4, seed=123, win_size=32):
    """
    固定正方形窗口（win_size×win_size）版 PatchMatch-KNN
    """
    win_radius = win_size // 2

    offsets, dists = initialize_aknn_numba_window(img, K, patch_size, win_radius, seed)

    H, W = img.shape
    base_radius = win_radius  # 更合理：抖动尺度不超过窗口半径

    for it in trange(iterations, desc="AKNN Iter"):
        propagation_step_numba_window(img, offsets, dists, patch_size, win_radius, it)

        radius = base_radius * (0.5 ** it)
        if radius < 1.0:
            radius = 1.0

        random_search_step_numba_window(
            img, offsets, dists, patch_size, win_radius, radius, seed + it * 10007
        )

    return offsets, dists


# =========================
# 7) Demo main
# =========================
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "data"
    noisy_path = DATA_DIR / "classic_photo_AWGN_sigma20_seed123456" / "lena_gray_sigma20_seed123456.png"

    noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
    if noisy_img is None:
        raise FileNotFoundError(f"找不到图片：{noisy_path}")

    img_float = noisy_img.astype(np.float32) / 255.0

    K = 4
    patch_size = 7
    iterations = 3
    seed = 123
    win_size = 32  # ✅ 只在 32x32 内搜索

    # 计时：注意第一次包含 JIT 编译时间
    t0 = time.perf_counter()
    final_offsets, final_dists = run_aknn_patchmatch_numba_window(
        img_float, K=K, patch_size=patch_size, iterations=iterations, seed=seed, win_size=win_size
    )
    t1 = time.perf_counter()
    print(f"[WITH JIT] run_aknn_patchmatch_numba_window took {t1 - t0:.3f} s")

    # 热身后再测纯运行
    _ = run_aknn_patchmatch_numba_window(
        img_float, K=K, patch_size=patch_size, iterations=1, seed=seed, win_size=win_size
    )
    t2 = time.perf_counter()
    _ = run_aknn_patchmatch_numba_window(
        img_float, K=K, patch_size=patch_size, iterations=iterations, seed=seed, win_size=win_size
    )
    t3 = time.perf_counter()
    print(f"[NO JIT ] run_aknn_patchmatch_numba_window took {t3 - t2:.3f} s")

    print("offsets:", final_offsets.shape, final_offsets.dtype)
    print("dists  :", final_dists.shape, final_dists.dtype)

    # 看几个点
    visualize_pixel_and_candidates(img_float, 32, 32, final_offsets, patch_size, title="Windowed PatchMatch (32,32)")
    visualize_pixel_and_candidates(img_float, 64, 64, final_offsets, patch_size, title="Windowed PatchMatch (64,64)")
    visualize_pixel_and_candidates(img_float, 280, 128, final_offsets, patch_size, title="Windowed PatchMatch (280,128)")

    # best dist 图
    best = final_dists[:, :, 0]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(img_float, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Best SSD (k=0)")
    plt.imshow(best, cmap="viridis")
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.show()
