import numpy as np
import cv2
import os
from pathlib import Path
# 添加 utils 文件夹到 Python 路径
from utils import getMetrics, AI_Metrics
import time

def gaussian_filter_custom(image, sigma):
    """
    实现了手动的高斯滤波 (Gaussian Filter)
    逻辑与之前的双边滤波一致，但只考虑空间距离权重 (Spatial Weights)，
    不考虑像素值差异 (Range/Color Weights)。

    参数:
    - image: 输入图像 (float)
    - sigma: 高斯核的标准差 (控制模糊程度)
    """
    # 1. 获取图像尺寸
    height, width = image.shape

    # 2. 转换浮点数
    img_float = image.astype(np.float64)
    output = np.zeros_like(img_float)

    # 3. 确定半径 (Radius)
    # 根据 3-sigma 准则确定窗口大小
    rho = int(np.ceil(3 * sigma))
    # 窗口直径
    k_size = 2 * rho + 1

    # 4. 边缘填充 (Padding)
    # 使用 Reflect 模式，与论文假设一致
    padded_img = cv2.copyMakeBorder(img_float, rho, rho, rho, rho, cv2.BORDER_REFLECT)

    # ============================================================
    # 预计算高斯核 (Spatial Kernel)
    # 这完全对应双边滤波里的 "spatial_weights" 部分
    # ============================================================
    sigma2 = sigma ** 2

    # 生成网格坐标 [-rho, ..., 0, ..., rho]
    y_idx, x_idx = np.meshgrid(np.arange(-rho, rho + 1), np.arange(-rho, rho + 1))

    # 计算距离平方
    dist_sq = x_idx ** 2 + y_idx ** 2

    # 计算高斯权重
    kernel = np.exp(-dist_sq / (2 * sigma2))

    # 归一化 Kernel (这一点非常重要，保证亮度不改变)
    # 也就是公式里的 1/C(x)
    kernel = kernel / np.sum(kernel)

    print(f"开始高斯滤波 (尺寸: {width}x{height}, sigma={sigma}, rho={rho})...")

    # 5. 遍历每一个像素
    for y in range(height):
        for x in range(width):
            # 提取邻域窗口 (Window)
            window = padded_img[y:y + k_size, x:x + k_size]

            # 卷积操作 (Convolution)
            # 对应公式: sum( u(y) * kernel(x-y) )
            weighted_sum = np.sum(window * kernel)

            output[y, x] = weighted_sum

    return np.clip(output, 0, 255).astype(np.uint8)


def get_method_noise(noisy_img, denoised_img):
    """
    计算 Method Noise (方法噪声)
    Method Noise = 原始噪图 - 去噪后的图
    它可以让我们看到"被当做噪声减掉的东西"里是不是包含了纹理细节。
    """
    # 转为 float 计算，并在加 128 以便显示 (将 0 均值移到灰色)
    diff = noisy_img.astype(np.float32) - denoised_img.astype(np.float32)
    # 归一化以便观察 (对比度拉伸)
    # 通常 method noise 很小，直接看是全黑的，需要放大显示
    return np.clip(diff + 128, 0, 255).astype(np.uint8)


# ==========================================
# 主程序
# ==========================================

# 1. 路径设置
original_path = "../../../data/classic_photo/lena_gray.png"
noisy_path = "../../../data/classic_photo_AWGN_sigma20_seed123456/lena_gray_sigma20_seed123456.png"

# 2. 读取图片
original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)

if original is None or noisy_img is None:
    print("错误：找不到图片，请检查路径。")
else:
    # 转 float
    original = original.astype(np.float32)
    noisy_img_float = noisy_img.astype(np.float32)

    # -------------------------------------------------
    # 3. 设置参数
    # -------------------------------------------------
    sigma_val = 0.9  # 高斯模糊的强度，你可以试试 1.0, 2.0, 3.0

    # 4. 运行高斯滤波
    start_time = time.perf_counter()
    denoised_img = gaussian_filter_custom(noisy_img_float, sigma=sigma_val)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"处理耗时: {elapsed_time:.4f} 秒")
    # 5. 计算 Method Noise
    method_noise_img = get_method_noise(noisy_img, denoised_img)

    # 6. 保存路径
    output_dir = '../../../out/images/baseline/gaussian_filter/'
    os.makedirs(output_dir, exist_ok=True)  # 确保文件夹存在

    out_name = f'{output_dir}lena_gray_gaussian_sigma{sigma_val}.png'
    noise_name = f'{output_dir}lena_gray_gaussian_method_noise_sigma{sigma_val}.png'

    cv2.imwrite(out_name, denoised_img)
    cv2.imwrite(noise_name, method_noise_img)

    # 7. 计算指标
    print("-" * 30)
    # 原噪图指标
    noise_metrics = getMetrics.calculate_metrics(original.astype(np.uint8), noisy_img.astype(np.uint8))
    print(f"【原带噪图】 PSNR: {noise_metrics['PSNR']:.2f} | SSIM: {noise_metrics['SSIM']:.4f}")

    # 去噪后指标
    denoised_metrics = getMetrics.calculate_metrics(original.astype(np.uint8), denoised_img)
    print(f"【高斯滤波】 PSNR: {denoised_metrics['PSNR']:.2f} | SSIM: {denoised_metrics['SSIM']:.4f}")
    print("-" * 30)
    print(f"处理完成！\n去噪图: {out_name}\nMethod Noise: {noise_name}")

    lpips, _ = AI_Metrics.compare_advanced_metrics(original_path, str(out_name))
    print(f"{lpips:.4f} ")