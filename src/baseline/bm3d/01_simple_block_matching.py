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

patch_size = 7
ref_y = 280
ref_x = 128
top_k = 16

# =========================
# Step 2 - Load
# =========================
noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
if noisy_img is None:
    raise FileNotFoundError(f"找不到图片：{noisy_path}")

noisy_img_float = noisy_img.astype(np.float32) / 255.0
H, W = noisy_img_float.shape

ref_patch = noisy_img_float[ref_y:ref_y+patch_size, ref_x:ref_x+patch_size]

# =========================
# Step 3 - Brute force search + timing
# =========================
matches = []  # (dist, y, x)

t0 = time.perf_counter()

for y in range(0, H - patch_size + 1):
    for x in range(0, W - patch_size + 1):
        cand_patch = noisy_img_float[y:y+patch_size, x:x+patch_size]
        diff = ref_patch - cand_patch
        dist = np.mean(diff * diff)
        matches.append((dist, y, x))

matches.sort(key=lambda t: t[0])
top_matches = matches[:top_k]

t1 = time.perf_counter()

print("=" * 60)
print(f"Brute-force window-free search took {t1 - t0:.4f} s")
print("=" * 60)

for i, (d, y, x) in enumerate(top_matches):
    print(f"{i}: dist={d:.6e}, pos=({y},{x})")

# =========================
# Step 4 - Visualization (big image)
# =========================
vis = (noisy_img_float * 255).astype(np.uint8)
vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

# Reference patch: green
cv2.rectangle(
    vis,
    (ref_x, ref_y),
    (ref_x + patch_size - 1, ref_y + patch_size - 1),
    (0, 255, 0),
    2
)

# Top-K: red
for _, y, x in top_matches:
    cv2.rectangle(
        vis,
        (x, y),
        (x + patch_size - 1, y + patch_size - 1),
        (0, 0, 255),
        1
    )

plt.figure(figsize=(6, 6))
plt.title("Reference (green) & Top-K matches (red)")
plt.imshow(vis[..., ::-1])
plt.axis("off")
plt.show()
