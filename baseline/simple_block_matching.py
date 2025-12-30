from pathlib import Path
import time

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def read_gray_float01(path: Path) -> np.ndarray:
    img_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img_u8.astype(np.float32) / 255.0

# =========================
# Step 1 - CONFIG (只改这里)
# =========================
base = Path("../Datasets/DAVIS/480p/bus-Y")
out_path = Path("../output/baseline/simple_block_matching")
out_path.mkdir(parents=True, exist_ok=True)


gt_path = base / "ori_photo" / "1_Y.png"

noisy_dir = base / "ori_photo_20AWGN_123456"
t_path   = noisy_dir / "y_01_noise.png"

patch_size = 16

# 你指定的源 patch 左上角坐标
ref_y = 240
ref_x = 235
img = read_gray_float01(t_path)
ref_patch = img[ref_y:ref_y+patch_size, ref_x:ref_x+patch_size]
H, W = img.shape
top_k = 10

matches = []  # 每个元素：(dist, y, x)

for y in range(0, H - patch_size + 1):
    for x in range(0, W - patch_size + 1):
        cand_patch = img[y:y+patch_size, x:x+patch_size]

        # L2 距离（MSE 形式）
        diff = ref_patch - cand_patch
        dist = np.mean(diff * diff)

        matches.append((dist, y, x))


matches.sort(key=lambda t: t[0])
top_matches = matches[:top_k]

for i, (d, y, x) in enumerate(top_matches):
    print(f"{i}: dist={d:.6e}, pos=({y},{x})")



vis = (img * 255).astype(np.uint8)
vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

# reference patch（绿色）
cv2.rectangle(
    vis,
    (ref_x, ref_y),
    (ref_x + patch_size - 1, ref_y + patch_size - 1),
    (0, 255, 0),
    2
)

# Top-K matches（红色）
for _, y, x in top_matches:
    cv2.rectangle(
        vis,
        (x, y),
        (x + patch_size - 1, y + patch_size - 1),
        (0, 0, 255),
        1
    )

cv2.imwrite(f"{out_path}/block_matching_vis.png", vis)

patches = []
for _, y, x in top_matches:
    patches.append(img[y:y+patch_size, x:x+patch_size])

patches = np.stack(patches, axis=0)  # (K, p, p)

# 简单拼成一排
collage = np.hstack(patches)
collage_u8 = (collage * 255).astype(np.uint8)
cv2.imwrite(f"{out_path}/matched_patches.png", collage_u8)
