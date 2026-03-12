import os
import cv2
import numpy as np
import bm3d
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd


# -------------------------
# 读取 PNG 并返回 YUV
# -------------------------
def read_png_to_yuv(path, normalize=True):
    img = cv2.imread(path)
    if img is None:
        return None, None, None, None

    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = yuv[:, :, 0]
    cr = yuv[:, :, 1]  # 注意：OpenCV 中索引1是 Cr
    cb = yuv[:, :, 2]  # 索引2是 Cb

    if normalize:
        y = y.astype(np.float32) / 255.0
        cb = cb.astype(np.float32) / 255.0
        cr = cr.astype(np.float32) / 255.0

    return y, cb, cr, img


# -------------------------
# 添加噪声
# -------------------------
def add_awgn_noise(y, sigma=25, seed=0):
    rng = np.random.default_rng(seed)
    sigma_norm = sigma / 255.0
    noise = rng.normal(0, sigma_norm, y.shape)
    noisy_y = np.clip(y + noise, 0, 1)
    return noisy_y


# -------------------------
# 可视化并保存
# -------------------------
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

    noise = y_noise - y # 直接计算噪声（不放大）
    noise_v = to_bgr_uint8(noise)

    # 拼接：第一行彩色，第二行亮度/残差
    row1 = np.hstack([img_bgr, noisy_bgr, denoised_bgr, noise_v])
    row2 = np.hstack([y_orig_v, y_noisy_v, y_denoised_v, residual_v])
    final_canvas = np.vstack([row1, row2])

    # 写标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = [
        "Orig Color", "Noisy Color", "Denoised Color", "noisy",
        "Orig Y", "Noisy Y", "Denoised Y", "Residual (x10)"
    ]
    for i, text in enumerate(labels):
        tx = (i % 4) * w + 10
        ty = (i // 4) * h + 30
        cv2.putText(final_canvas, text, (tx, ty), font, 0.7, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(img_save_dir, f"full_process_{idx:02d}.png"), final_canvas)


# -------------------------
# 主循环
# -------------------------
results = []
dataset = "PhotoCD_PCD0992"
img_save_dir = f"out/images/bm3d_baseline_HARD_THRESHOLDING/{dataset}"
res_save_dir = "out/results"

# 关键修改：确保文件夹存在
os.makedirs(img_save_dir, exist_ok=True)
os.makedirs(res_save_dir, exist_ok=True)

for i in range(1, 25):
    image_path = f"data/{dataset}/{i:02d}.png"
    if not os.path.exists(image_path): continue

    y, cb, cr, img_bgr = read_png_to_yuv(image_path)

    sigma = 25
    sigma_norm = sigma / 255.0
    y_noise = add_awgn_noise(y, sigma)

    # BM3D 这里的 z 接受 [0,1] 范围
    denoised_y_norm = bm3d.bm3d(
        z=y_noise,
        sigma_psd=sigma_norm,
        stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
    )
    y_denoised = np.clip(denoised_y_norm, 0, 1)

    # 调用显示函数
    showPic(img_bgr, y, y_noise, cb, cr, y_denoised, img_save_dir, i)

    # 使用裁剪后的图像计算指标
    current_psnr = psnr(y, y_denoised, data_range=1.0)
    current_ssim = ssim(y, y_denoised, data_range=1.0)

    results.append({
        'No': f"{i:02d}",
        'Sigma': sigma,
        'PSNR': round(current_psnr, 2),
        'SSIM': round(current_ssim, 4)
    })
    print(f"PSNR: {current_psnr:.2f} dB | SSIM: {current_ssim:.4f}\n")

# 保存结果
df = pd.DataFrame(results)
csv_path = os.path.join(res_save_dir, f"bm3d_HARD_THRESHOLDING_results_{dataset}.csv")
df.to_csv(csv_path, index=False)
print(f"所有结果已成功保存到 {csv_path}")