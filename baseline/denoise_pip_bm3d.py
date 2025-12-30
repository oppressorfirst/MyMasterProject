import numpy as np
import cv2
from pathlib import Path
from bm3d import bm3d, BM3DStages
import time
from skimage.metrics import structural_similarity as ssim

def psnr(gt, x):
    mse = np.mean((gt - x) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)

def stats_str(name, img01):
    return (f"[STAT] {name}: shape={img01.shape}, dtype={img01.dtype}, "
            f"min={img01.min():.4f}, max={img01.max():.4f}, "
            f"mean={img01.mean():.4f}, std={img01.std():.4f}")

# =========================
# Step 1) 路径
# =========================
base = Path("../Datasets/DAVIS/480p/bus-Y")

gt_path    = base / "ori_photo" / "1_Y.png"
noisy_path = base / "ori_photo_20AWGN_123456" / "y_01_noise.png"

out_dir = base / ("bm3d_" + noisy_path.parent.name)
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "1_Y_bm3d.png"

# 新增：log 文件
log_path = out_dir / "run_log.txt"

def log_print(msg: str):
    # 同时输出到终端 + 写入log文件
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# 每次运行先写个分隔，避免多次运行混在一起
log_print("\n" + "="*60)
log_print(f"[RUN] {time.strftime('%Y-%m-%d %H:%M:%S')}")

# =========================
# Step 2) 读图
# =========================
gt_u8 = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
noisy_u8 = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)

if gt_u8 is None:
    raise FileNotFoundError(f"读不到GT: {gt_path}")
if noisy_u8 is None:
    raise FileNotFoundError(f"读不到Noisy: {noisy_path}")

# =========================
# Step 3) float [0,1]
# =========================
gt = gt_u8.astype(np.float32) / 255.0
noisy = noisy_u8.astype(np.float32) / 255.0

log_print(stats_str("GT", gt))
log_print(stats_str("Noisy", noisy))

# =========================
# Step 4) BM3D + 时间
# =========================
sigma_255 = 20
sigma = sigma_255 / 255.0

t0 = time.perf_counter()
den = bm3d(noisy, sigma_psd=sigma, stage_arg=BM3DStages.ALL_STAGES)
t1 = time.perf_counter()

log_print(stats_str("BM3D", den))
log_print(f"[TIME] BM3D runtime: {(t1 - t0)*1000:.2f} ms")

# =========================
# Step 5) PSNR + SSIM
# =========================
psnr_noisy = psnr(gt, noisy)
psnr_bm3d  = psnr(gt, den)

ssim_noisy = ssim(gt, noisy, data_range=1.0)
ssim_bm3d  = ssim(gt, den,   data_range=1.0)

log_print(f"[INFO] GT     : {gt_path}")
log_print(f"[INFO] Noisy  : {noisy_path}")
log_print(f"[INFO] OutDir : {out_dir}")
log_print(f"[INFO] Log    : {log_path}")

log_print(f"Noisy PSNR: {psnr_noisy:.2f} dB | SSIM: {ssim_noisy:.4f}")
log_print(f"BM3D  PSNR: {psnr_bm3d:.2f} dB | SSIM: {ssim_bm3d:.4f}")

# =========================
# Step 6) 存结果（顺便把 den clip 一下更干净）
# =========================
den_clip = np.clip(den, 0.0, 1.0)
cv2.imwrite(str(out_path), (den_clip * 255.0 + 0.5).astype(np.uint8))
log_print(f"[OK] Saved: {out_path}")
