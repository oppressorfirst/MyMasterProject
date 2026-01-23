from pathlib import Path
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
noisy_path = DATA_DIR / "classic_photo_AWGN_sigma20_seed123456" / "lena_gray_sigma20_seed123456.png"

patch_size = 8
top_k = 16
window_size = 39
stride = patch_size // 2
search_radius = window_size // 2

# =========================
# Numba 核心
# =========================
@njit
def patch_mse(img, y1, x1, y2, x2, patch_size):
    s = 0.0
    for i in range(patch_size):
        for j in range(patch_size):
            d = img[y1 + i, x1 + j] - img[y2 + i, x2 + j]
            s += d * d
    return s / (patch_size * patch_size)


@njit
def window_match_single(img, ref_y, ref_x, patch_size, top_k, search_radius):
    H, W = img.shape

    dists = np.full(top_k, 1e9, dtype=np.float32)
    ys = np.zeros(top_k, dtype=np.int32)
    xs = np.zeros(top_k, dtype=np.int32)

    y_min = max(0, ref_y - search_radius)
    y_max = min(H - patch_size, ref_y + search_radius)
    x_min = max(0, ref_x - search_radius)
    x_max = min(W - patch_size, ref_x + search_radius)

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if y == ref_y and x == ref_x:
                continue

            d = patch_mse(img, ref_y, ref_x, y, x, patch_size)

            if d < dists[top_k - 1]:
                pos = top_k - 1
                while pos > 0 and d < dists[pos - 1]:
                    dists[pos] = dists[pos - 1]
                    ys[pos] = ys[pos - 1]
                    xs[pos] = xs[pos - 1]
                    pos -= 1
                dists[pos] = d
                ys[pos] = y
                xs[pos] = x

    return dists, ys, xs

# =========================
# Load
# =========================
noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
if noisy_img is None:
    raise FileNotFoundError(f"找不到图片：{noisy_path}")

noisy_img_float = noisy_img.astype(np.float32) / 255.0
H, W = noisy_img_float.shape

# =========================
# Matching (Numba)
# =========================
all_results = []

ref_ys = list(range(0, H - patch_size + 1, stride))
ref_xs = list(range(0, W - patch_size + 1, stride))
total = len(ref_ys) * len(ref_xs)

# 触发 JIT 编译（第一次慢，后面飞快）
print("Warming up Numba JIT...")
_ = window_match_single(noisy_img_float, ref_ys[0], ref_xs[0],
                        patch_size, top_k, search_radius)

t_global0 = time.perf_counter()
pbar = tqdm(total=total, desc="Window matching (Numba)", dynamic_ncols=True)

for ref_y in ref_ys:
    for ref_x in ref_xs:
        t0 = time.perf_counter()

        dists, ys, xs = window_match_single(
            noisy_img_float,
            ref_y,
            ref_x,
            patch_size,
            top_k,
            search_radius
        )

        t1 = time.perf_counter()

        top_matches = [(float(dists[i]), int(ys[i]), int(xs[i])) for i in range(top_k)]

        all_results.append({
            "ref_pos": (ref_y, ref_x),
            "top_matches": top_matches,
            "time": t1 - t0
        })

        pbar.update(1)

        if (pbar.n % 200) == 0 or pbar.n == total:
            elapsed = time.perf_counter() - t_global0
            avg_time = elapsed / pbar.n
            pbar.set_postfix({
                "elapsed_s": f"{elapsed:.1f}",
                "avg_s/ref": f"{avg_time:.5f}",
            })

pbar.close()
t_global1 = time.perf_counter()

print("=" * 60)
print(f"Total windowed matching time (Numba): {t_global1 - t_global0:.3f} s")
print("=" * 60)

# =========================
# Visualization
# =========================
idx = 556
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

plt.figure(figsize=(6, 6))
plt.title(f"Ref {idx} (green=ref, red=topK)")
plt.imshow(vis[..., ::-1])
plt.axis("off")
plt.show()
