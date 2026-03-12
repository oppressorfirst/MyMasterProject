from pathlib import Path
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Step 1 - CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
noisy_path = DATA_DIR / "classic_photo_AWGN_sigma20_seed123456" / "lena_gray_sigma20_seed123456.png"

patch_size = 8
top_k = 16
window_size = 39

stride = patch_size// 2  # 不重叠
search_radius = window_size // 2

# =========================
# Step 2 - Load
# =========================

noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
if noisy_img is None:
    raise FileNotFoundError(f"找不到图片：{noisy_path}")

noisy_img_float = noisy_img.astype(np.float32) / 255.0
H, W = noisy_img_float.shape

# =========================
# Step 3 - Brute-force windowed matching + timing
# =========================
from tqdm import tqdm
import time

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

for ref_y in ref_ys:
    for ref_x in ref_xs:
        t0 = time.perf_counter()

        ref_patch = noisy_img_float[
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

                cand_patch = noisy_img_float[y:y + patch_size, x:x + patch_size]
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
idx = 556  # 你原来选的
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


