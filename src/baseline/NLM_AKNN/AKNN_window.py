import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm, trange
from pathlib import Path
import matplotlib.patches as patches
import time


# =========================
# 1) Patch 距离（SSD）
# =========================
def compute_patch_distance(img, y, x, ny, nx, patch_size):
    """
    计算两个 patch 的 SSD（越界返回 inf）
    img: (H,W) float
    """
    H, W = img.shape
    r = patch_size // 2

    # 源 patch 必须完整
    if y - r < 0 or y + r >= H or x - r < 0 or x + r >= W:
        return float("inf")
    # 目标 patch 必须完整
    if ny - r < 0 or ny + r >= H or nx - r < 0 or nx + r >= W:
        return float("inf")

    patch_src = img[y - r: y + r + 1, x - r: x + r + 1]
    patch_tgt = img[ny - r: ny + r + 1, nx - r: nx + r + 1]
    diff = patch_src - patch_tgt
    return float(np.sum(diff * diff))


# =========================
# 2) Window 初始化（只在 win_size×win_size 内采样）
# =========================
def initialize_aknn_window(img, K, patch_size=7, win_size=16):
    """
    Window 版初始化：每个像素只在局部窗口内随机采样 K 个候选
    offsets: (H,W,K,2) (dy,dx)
    dists  : (H,W,K)
    """
    H, W = img.shape
    r = patch_size // 2
    win_radius = win_size // 2

    nn_offsets = np.zeros((H, W, K, 2), dtype=np.int32)
    nn_dists = np.full((H, W, K), float("inf"), dtype=np.float32)

    print(f"Initializing WINDOW AKNN: HxW={H}x{W}, K={K}, patch={patch_size}, window={win_size}x{win_size}")

    for y in tqdm(range(H), desc="Init", leave=False):
        for x in range(W):
            # 源 patch 不完整：跳过
            if y - r < 0 or y + r >= H or x - r < 0 or x + r >= W:
                continue

            candidates = []
            tries = 0
            # 多试一些，避免边界附近采不到足够候选
            while len(candidates) < K and tries < K * 50:
                tries += 1

                dy = np.random.randint(-win_radius, win_radius + 1)
                dx = np.random.randint(-win_radius, win_radius + 1)

                # 排除 self
                if dy == 0 and dx == 0:
                    continue

                ny, nx = y + dy, x + dx

                # 目标 patch 不完整：丢弃
                if ny - r < 0 or ny + r >= H or nx - r < 0 or nx + r >= W:
                    continue

                dist = compute_patch_distance(img, y, x, ny, nx, patch_size)
                candidates.append((dist, dy, dx))

            candidates.sort(key=lambda t: t[0])
            m = min(K, len(candidates))
            for k in range(m):
                nn_dists[y, x, k] = candidates[k][0]
                nn_offsets[y, x, k, 0] = candidates[k][1]
                nn_offsets[y, x, k, 1] = candidates[k][2]

    return nn_offsets, nn_dists


# =========================
# 3) Top-K 插入（维护升序列表）
# =========================
def _insert_topk_python(new_dist, new_dy, new_dx, dists_1px, offs_1px):
    """
    dists_1px: (K,), offs_1px: (K,2)
    """
    K = dists_1px.shape[0]

    # 不如最差的就丢
    if new_dist >= dists_1px[K - 1]:
        return

    # 查重
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

    # 右移
    for t in range(K - 1, pos, -1):
        dists_1px[t] = dists_1px[t - 1]
        offs_1px[t, 0] = offs_1px[t - 1, 0]
        offs_1px[t, 1] = offs_1px[t - 1, 1]

    dists_1px[pos] = new_dist
    offs_1px[pos, 0] = new_dy
    offs_1px[pos, 1] = new_dx


# =========================
# 4) 更新某像素的 top-K（带窗口约束）
# =========================
def update_best_k_window(img, y, x, prop_dy, prop_dx,
                         offsets, dists, patch_size, win_size):
    """
    尝试将 (prop_dy, prop_dx) 插入 (y,x) 的 top-K
    """
    H, W = img.shape
    r = patch_size // 2
    win_radius = win_size // 2

    ny, nx = y + prop_dy, x + prop_dx

    # window 约束：目标必须在窗口内
    if abs(ny - y) > win_radius or abs(nx - x) > win_radius:
        return

    # patch 完整性约束
    if y - r < 0 or y + r >= H or x - r < 0 or x + r >= W:
        return
    if ny - r < 0 or ny + r >= H or nx - r < 0 or nx + r >= W:
        return

    new_dist = compute_patch_distance(img, y, x, ny, nx, patch_size)
    _insert_topk_python(new_dist, prop_dy, prop_dx, dists[y, x], offsets[y, x])


# =========================
# 5) 传播（Propagation）—— 必须保持扫描顺序（有“抄作业”依赖）
# =========================
def propagation_step_window(img, offsets, dists, patch_size, win_size, iter_num):
    H, W = img.shape
    K = offsets.shape[2]
    r = patch_size // 2

    if iter_num % 2 == 0:
        # 正向：上、左
        y_range = range(r, H - r)
        x_range = range(r, W - r)
        neighbor_deltas = [(-1, 0), (0, -1)]
        desc = f"Prop fwd {iter_num+1}"
    else:
        # 反向：下、右
        y_range = range(H - r - 1, r - 1, -1)
        x_range = range(W - r - 1, r - 1, -1)
        neighbor_deltas = [(1, 0), (0, 1)]
        desc = f"Prop rev {iter_num+1}"

    for y in tqdm(y_range, desc=desc, leave=False):
        for x in x_range:
            for dy_n, dx_n in neighbor_deltas:
                nb_y, nb_x = y + dy_n, x + dx_n
                nb_offsets = offsets[nb_y, nb_x]  # (K,2)

                for k in range(K):
                    prop_dy = int(nb_offsets[k, 0])
                    prop_dx = int(nb_offsets[k, 1])
                    update_best_k_window(img, y, x, prop_dy, prop_dx,
                                         offsets, dists, patch_size, win_size)


# =========================
# 6) 随机搜索（Random Search）—— 也必须窗口约束
# =========================
def random_search_step_window(img, offsets, dists, patch_size, win_size, radius):
    H, W = img.shape
    K = offsets.shape[2]
    r = patch_size // 2
    win_radius = win_size // 2

    for y in tqdm(range(r, H - r), desc="Random", leave=False):
        for x in range(r, W - r):
            for k in range(K):
                best_dy = int(offsets[y, x, k, 0])
                best_dx = int(offsets[y, x, k, 1])

                rd_y = int(round(radius * np.random.randn()))
                rd_x = int(round(radius * np.random.randn()))

                cand_dy = best_dy + rd_y
                cand_dx = best_dx + rd_x

                # 强制 clamp 到 window 内（避免反复 continue）
                cand_dy = int(np.clip(cand_dy, -win_radius, win_radius))
                cand_dx = int(np.clip(cand_dx, -win_radius, win_radius))

                if cand_dy == 0 and cand_dx == 0:
                    continue

                update_best_k_window(img, y, x, cand_dy, cand_dx,
                                     offsets, dists, patch_size, win_size)


# =========================
# 7) 主循环（Window AKNN PatchMatch）
# =========================
def run_aknn_window(img, K=4, patch_size=8, win_size=16, iterations=3, seed=123):
    np.random.seed(seed)

    # init
    offsets, dists = initialize_aknn_window(img, K=K, patch_size=patch_size, win_size=win_size)

    # radius：你也可以设置成 win_radius，更符合“只在窗内”
    win_radius = win_size // 2
    base_radius = float(win_radius)

    for it in trange(iterations, desc="AKNN Iter"):
        t0 = time.perf_counter()

        propagation_step_window(img, offsets, dists, patch_size, win_size, it)

        radius = base_radius * (0.5 ** it)
        if radius < 1.0:
            radius = 1.0

        random_search_step_window(img, offsets, dists, patch_size, win_size, radius)

        t1 = time.perf_counter()
        tqdm.write(f"iter {it+1}/{iterations}: {t1 - t0:.3f}s")

    return offsets, dists


# =========================
# 8) 可视化（红=中心patch，蓝=候选patch）
# =========================
def visualize_pixel_and_candidates(img, y0, x0, offsets, patch_size, title=None):
    H, W = img.shape
    K = offsets.shape[2]
    r = patch_size // 2

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.set_title(title if title else f"Pixel ({y0}, {x0}) and its {K} candidates")
    ax.axis("off")

    # 中心 patch（红）
    ax.add_patch(
        patches.Rectangle((x0 - r, y0 - r), patch_size, patch_size,
                          linewidth=2, edgecolor="red", facecolor="none")
    )

    # 候选 patch（蓝）
    for k in range(K):
        dy = int(offsets[y0, x0, k, 0])
        dx = int(offsets[y0, x0, k, 1])
        ny, nx = y0 + dy, x0 + dx

        if 0 <= ny < H and 0 <= nx < W:
            ax.add_patch(
                patches.Rectangle((nx - r, ny - r), patch_size, patch_size,
                                  linewidth=1.5, edgecolor="blue", facecolor="none")
            )
            ax.text(nx, ny, f"{k}", color="blue", fontsize=10)

    plt.show()


# =========================
# 9) main
# =========================
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "data"
    noisy_path = DATA_DIR / "classic_photo_AWGN_sigma20_seed123456" / "lena_gray_sigma20_seed123456.png"

    noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
    if noisy_img is None:
        raise FileNotFoundError(f"找不到图片：{noisy_path}")

    img_float = noisy_img.astype(np.float32) / 255.0

    K = 16
    patch_size = 8
    iterations = 3
    win_size = 39
    seed = 123

    t0 = time.perf_counter()
    final_offsets, final_dists = run_aknn_window(
        img_float, K=K, patch_size=patch_size, win_size=win_size, iterations=iterations, seed=seed
    )
    t1 = time.perf_counter()

    print(f"run_aknn_window took {t1 - t0:.3f} s")
    print("offsets:", final_offsets.shape, final_offsets.dtype)
    print("dists  :", final_dists.shape, final_dists.dtype)

    # 看几个点
    visualize_pixel_and_candidates(img_float, 32, 32, final_offsets, patch_size, title="Window PatchMatch (32,32)")
    visualize_pixel_and_candidates(img_float, 64, 64, final_offsets, patch_size, title="Window PatchMatch (64,64)")
    visualize_pixel_and_candidates(img_float, 280, 128, final_offsets, patch_size, title="Window PatchMatch (280,128)")
