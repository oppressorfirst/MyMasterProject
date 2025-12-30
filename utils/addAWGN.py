import cv2
import numpy as np
from pathlib import Path
import re

# ====== 配置区：改这里就行 ======
sigma = 20
in_dir = Path("../Datasets/DAVIS/480p/bus-Y/ori_photo")   # 你的输入文件夹（改成你的实际路径）
seed = 123456
# ===============================

out_dir = in_dir.parent / f"noise_photo_{sigma}AWGN_{seed}"
out_dir.mkdir(parents=True, exist_ok=True)

# 支持的图片后缀（按需增减）
exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# 可选：固定随机种子，保证每次运行结果一致（论文/对比更稳）
np.random.seed(seed)

cnt = 0
for p in sorted(in_dir.iterdir()):
    if not p.is_file() or p.suffix.lower() not in exts:
        continue

    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Skip (read fail): {p.name}")
        continue

    img_f = img.astype(np.float32)
    noise = np.random.normal(0.0, sigma, size=img_f.shape).astype(np.float32)
    noisy = np.clip(img_f + noise, 0, 255).astype(np.uint8)

    m = re.match(r"(\d+)_Y", p.stem)  # p.stem: "1_Y"
    if m:
        n = int(m.group(1))
        out_name = f"y_{n:02d}_noise.png"
    else:
        out_name = f"{p.stem}_noise.png"

    out_path = out_dir / out_name
    cv2.imwrite(str(out_path), noisy)
    cnt += 1

print(f"Done: {cnt} images -> {out_dir}")
