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
this_file = Path(__file__).resolve()
base = Path("../Datasets/DAVIS/480p/bus-Y")
out_path = Path(f"../output/baseline/{this_file.stem}")
out_path.mkdir(parents=True, exist_ok=True)


img_gt_path = base / "ori_photo" / "1_Y.png"

noisy_dir = base / "noise_photo_20AWGN_123456"
img_noise_path   = noisy_dir / "y_01_noise.png"

patch_size = 16
stride = patch_size  # 不重叠
img = read_gray_float01(img_noise_path)
H, W = img.shape
top_k = 10



# 每个元素是一个 dict，表示一个 reference patch 的结果
all_results = []



ref_id = 0

for ref_y in range(0, H - patch_size + 1, stride):
    for ref_x in range(0, W - patch_size + 1, stride):

        ref_patch = img[
            ref_y:ref_y + patch_size,
            ref_x:ref_x + patch_size
        ]

        matches = []  # (dist, y, x)
        for y in range(0, H - patch_size + 1):
            for x in range(0, W - patch_size + 1):
                cand_patch = img[y:y + patch_size, x:x + patch_size]

                diff = ref_patch - cand_patch
                dist = np.mean(diff * diff)

                matches.append((dist, y, x))
        matches.sort(key=lambda t: t[0])
        top_matches = matches[:top_k]
        all_results.append({
            "ref_id": ref_id,
            "ref_pos": (ref_y, ref_x),
            "top_matches": top_matches,
        })

        print(
            f"[{ref_id:04d}] ref=({ref_y},{ref_x}) "
            f"best_dist={top_matches[1][0]:.3e}"
        )

        ref_id += 1


idx = 10  # 随便挑一个
res = all_results[idx]

ref_y, ref_x = res["ref_pos"]
top_matches = res["top_matches"]

vis = (img * 255).astype(np.uint8)
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

cv2.imwrite(f"{out_path}/block_matching_vis.png", vis)

patches = [
    img[y:y+patch_size, x:x+patch_size]
    for _, y, x in top_matches
]

collage = np.hstack(patches)
cv2.imwrite(
    f"{out_path}/matched_patches.png",
    (collage * 255).astype(np.uint8)
)
