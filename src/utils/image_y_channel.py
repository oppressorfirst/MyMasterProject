import cv2
import numpy as np
from pathlib import Path

# ====== 你只需要改这两个 ======
in_dir = Path("../Datasets/DAVIS/480p/bus")        # 输入文件夹
out_dir = Path("../Datasets/DAVIS/480p/bus-Y")    # 输出文件夹
# =================================

out_dir.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

cnt = 0
for p in in_dir.iterdir():
    if not p.is_file():
        continue
    if p.suffix.lower() not in exts:
        continue

    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        print("[FAIL] cannot read:", p)
        continue

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0]  # (H, W) uint8

    out_path = out_dir / f"{p.stem}_Y.png"
    ok = cv2.imwrite(str(out_path), Y)
    if ok:
        cnt += 1
        print("[OK]", p.name, "->", out_path.name)
    else:
        print("[FAIL] write:", out_path)

print("Done. Converted:", cnt)
