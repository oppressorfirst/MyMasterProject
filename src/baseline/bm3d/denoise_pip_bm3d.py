import numpy as np
import cv2
from pathlib import Path
from bm3d import bm3d, BM3DStages
import time
from skimage.metrics import structural_similarity as ssim
from src.utils import getMetrics, AI_Metrics


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
# 定位到项目根 MyMasterProject
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "out"

gt_path    = DATA_DIR / "classic_photo" / "lena_gray.png"
noisy_path =  DATA_DIR / "classic_photo_AWGN_sigma20_seed123456" / "lena_gray_sigma20_seed123456.png"

out_dir =  OUT_DIR/ "images"/ "baseline"/"bm3d"
out_dir.mkdir(parents=True, exist_ok=True)


sigma_255 = 12
sigma = sigma_255 / 255.0

out_path = out_dir / f"lena_pip_bm3d_sigma{sigma:.2f}.png"

# 新增：log 文件
log_path = out_dir / "pip_bm3d_run_log.txt"

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


t0 = time.perf_counter()
den = bm3d(noisy, sigma_psd=sigma, stage_arg=BM3DStages.ALL_STAGES)
t1 = time.perf_counter()
den_clip = np.clip(den, 0.0, 1.0)
den_u8 = (den_clip * 255.0 + 0.5).astype(np.uint8)
cv2.imwrite(str(out_path), den_u8)
log_print(f"[OK] Saved: {out_path}")

print(f"[TIME] BM3D runtime: {(t1 - t0)*1000:.2f} ms")
print("-" * 30)
noise_metrics = getMetrics.calculate_metrics(gt_u8.astype(np.uint8), noisy_u8.astype(np.uint8))
print(f"【原带噪图】 PSNR: {noise_metrics['PSNR']:.2f} | SSIM: {noise_metrics['SSIM']:.4f}")

# 7. 计算指标 (Denoised vs Original)
denoised_metrics = getMetrics.calculate_metrics(gt_u8.astype(np.uint8), den_u8)
print(f"【去噪声后】 PSNR: {denoised_metrics['PSNR']:.2f} | SSIM: {denoised_metrics['SSIM']:.4f}")
print("-" * 30)
print(f"处理完成！图片已保存为: {out_path}")

lpips, _ = AI_Metrics.compare_advanced_metrics(str(gt_path), str(out_path))
print(f"{lpips:.4f} ")
