import cv2
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import os
import pathlib

# -------------------------
# 读取 PNG 并返回 YUV 通道
# -------------------------
def read_png_to_yuv(path):
    img = cv2.imread(path)
    if img is None:
        return None, None, None, None

    # 转换为 YCrCb 空间
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    y = yuv[:, :, 0]
    cr = yuv[:, :, 1]
    cb = yuv[:, :, 2]
    return y, cr, cb, img  # 同时返回原图用于对比

def showPic(img_bgr, y, y_noise, cb, cr, y_denoised, img_save_dir, idx):
    def to_bgr_uint8(img_gray):
        gray_255 = (np.clip(img_gray, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(gray_255, cv2.COLOR_GRAY2BGR)

    h, w, _ = img_bgr.shape

    # 合成彩色图 (顺序必须是 Y, Cr, Cb)
    noisy_yuv = np.stack([(y_noise * 255).astype(np.uint8),
                          (cr * 255).astype(np.uint8),
                          (cb * 255).astype(np.uint8)], axis=2)
    noisy_bgr = cv2.cvtColor(noisy_yuv, cv2.COLOR_YCrCb2BGR)

    denoised_yuv = np.stack([(y_denoised * 255).astype(np.uint8),
                             (cr * 255).astype(np.uint8),
                             (cb * 255).astype(np.uint8)], axis=2)
    denoised_bgr = cv2.cvtColor(denoised_yuv, cv2.COLOR_YCrCb2BGR)

    # 计算残差 (降噪去掉的部分)
    residual = np.abs(y_denoised - y)
    residual_vis = np.clip(residual * 5, 0, 1)  # 放大5倍

    # 转换各通道用于拼接
    y_orig_v = to_bgr_uint8(y)
    y_noisy_v = to_bgr_uint8(y_noise)  # 修复这里：之前误写成了 y
    y_denoised_v = to_bgr_uint8(y_denoised)
    residual_v = to_bgr_uint8(residual_vis)

    noise = y_noise - y
    noise_v = to_bgr_uint8(noise)

    # 拼接：第一行彩色，第二行亮度/残差
    row1 = np.hstack([img_bgr, noisy_bgr, denoised_bgr, noise_v])
    row2 = np.hstack([y_orig_v, y_noisy_v, y_denoised_v, residual_v])
    final_canvas = np.vstack([row1, row2])

    # 写标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        "Orig Color", "Noisy Color", "Denoised Color", "noisy ",
        "Orig Y", "Noisy Y", "Denoised Y", "Residual (x5)"
    ]
    for i, text in enumerate(labels):
        tx = (i % 4) * w + 10
        ty = (i // 4) * h + 30
        cv2.putText(final_canvas, text, (tx, ty), font, 0.7, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(img_save_dir, f"full_process_{idx:02d}.png"), final_canvas)


# -------------------------
# 主循环逻辑
# -------------------------


dataset = "PhotoCD_PCD0992"
output_dir = f"out/images/nlm_baseline/{dataset}"
img_save_dir = f"out/images/nlm_baseline/{dataset}"
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

results = []
sigma_val = 25

for i in range(1, 25):
    img_name = f"{i:02d}.png"
    image_path = f"data/{dataset}/{img_name}"

    # 1. 读取数据 (y=亮度, cr=红色差, cb=蓝色差)
    y, cr, cb, original_bgr = read_png_to_yuv(image_path)
    if y is None: continue

    # 2. 模拟噪声 (仅对 Y 通道加噪或对全通道加噪，这里演示对 Y 加噪)
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, sigma_val / 255.0, y.shape)
    y_noisy = np.clip(y + noise, 0, 1).astype(np.float32)

    # 3. NLM 去噪
    print(f"[{i:02d}/24] Denoising Y channel...")
    sigma_est = np.mean(estimate_sigma(y_noisy, channel_axis=None))
    denoised_y = denoise_nl_means(
        y_noisy,
        h=1.15 * (sigma_val / 255.0),
        fast_mode=True,
        patch_size=5,
        patch_distance=6,
        channel_axis=None
    )
    denoised_y = np.clip(denoised_y, 0, 1).astype(np.float32)

    showPic(original_bgr, y, y_noisy, cb, cr, denoised_y, img_save_dir, i)
    # 4. 【核心步骤】合成彩色图
    # 将去噪后的 Y 与 原始 Cr, Cb 重新组合
    yuv_recombined = np.stack([denoised_y, cr, cb], axis=2)
    # 将 0-1 范围转回 0-255 uint8
    yuv_uint8 = (yuv_recombined * 255.0).astype(np.uint8)
    # 色彩空间转回 BGR
    denoised_bgr = cv2.cvtColor(yuv_uint8, cv2.COLOR_YCrCb2BGR)

    # 5. 计算指标 (通常在 Y 通道计算 PSNR 更有意义)
    cur_psnr = psnr(y, denoised_y, data_range=1.0)
    cur_ssim = ssim(y, denoised_y, data_range=1.0)

    results.append({
        'No': i,
        'PSNR': round(cur_psnr, 2),
        'SSIM': round(cur_ssim, 4)
    })



    print(f"Done No: {i:02d} | PSNR: {cur_psnr:.2f}")

# 保存 CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, "nlm_color_results.csv"), index=False)
print("所有彩色结果已保存！")