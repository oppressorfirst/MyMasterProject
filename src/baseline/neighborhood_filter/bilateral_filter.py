import numpy as np
import cv2

# 添加 utils 文件夹到 Python 路径
from utils import getMetrics, AI_Metrics


def bilateral_filter_custom(image, rho, h, sigma_s):
    """
    实现了双边滤波 (Bilateral Filter / SNF)
    对应公式: Weight = exp(-|I_p - I_q|^2 / h^2) * exp(-|p - q|^2 / sigma_s^2)

    参数:
    - image: 输入图像
    - rho: 邻域半径 (决定窗口大小为 2*rho+1)
    - h: 颜色/灰度 差异的敏感度 (对应公式中的 h)
    - sigma_s: 空间距离的敏感度 (对应你截图中公式里的 ρ)

    注意:
    - 如果 sigma_s 很大，空间权重就接近1，这就退化成了 YNF。
    - 如果 h 很大，颜色权重就接近1，这就退化成了普通的高斯模糊。
    """
    # 1. 获取图像尺寸
    height, width = image.shape

    # 2. 转换浮点数
    img_float = image.astype(np.float64)
    output = np.zeros_like(img_float)

    # 3. 边缘填充
    padded_img = cv2.copyMakeBorder(img_float, rho, rho, rho, rho, cv2.BORDER_REFLECT)

    # 预先计算常数
    h2 = h ** 2
    sigma_s2 = sigma_s ** 2

    # ============================================================
    # 关键优化：预计算“空间权重模板” (Spatial Weights)
    # 因为不管窗口移到哪里，像素点之间的相对距离是不变的。
    # ============================================================
    # 生成一个坐标网格，范围从 -rho 到 +rho
    # 比如 rho=1, 结果就是 [[-1, 0, 1], [-1, 0, 1], ...]
    y_idx, x_idx = np.meshgrid(np.arange(-rho, rho + 1), np.arange(-rho, rho + 1))

    # 计算距离平方: x^2 + y^2
    dist_sq = x_idx ** 2 + y_idx ** 2

    # 计算空间高斯权重 (这就是公式里新增的那一项 e^(-|x-y|^2 / rho^2))
    # 这个矩阵在整个循环中是通用的
    spatial_weights = np.exp(-dist_sq / sigma_s2)

    print(f"开始双边滤波 (尺寸: {width}x{height})...")

    # 4. 遍历每一个像素
    for y in range(height):
        for x in range(width):
            center_val = padded_img[y + rho, x + rho]

            # 提取邻域窗口 (和 YNF 一样)
            window = padded_img[y:y + 2 * rho + 1, x:x + 2 * rho + 1]

            # --- 步骤 A: 计算颜色权重 (Range Kernel) ---
            # 这部分和 YNF 一模一样：看长得像不像
            color_diff_sq = (window - center_val) ** 2
            color_weights = np.exp(-color_diff_sq / h2)

            # --- 步骤 B: 结合空间权重 (Spatial Kernel) ---
            # 双边滤波的核心：总权重 = 颜色权重 * 空间权重
            total_weights = color_weights * spatial_weights

            # --- 步骤 C: 归一化并求和 ---
            normalization_factor = np.sum(total_weights)
            weighted_sum = np.sum(window * total_weights)

            output[y, x] = weighted_sum / normalization_factor

    return np.clip(output, 0, 255).astype(np.uint8)


# ==========================================
# 主程序
# ==========================================

# 1. 读取路径配置
original_path = "../../../data/classic_photo/lena_gray.png"
noisy_path = "../../../data/classic_photo_AWGN_sigma20_seed123456/lena_gray_sigma20_seed123456.png"

# 读取图片
original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)

# 确保图片读取成功
if original is None or noisy_img is None:
    print("错误：找不到图片，请检查路径。")
else:
    original = original.astype(np.float32)
    noisy_img_float = noisy_img.astype(np.float32)

    # -------------------------------------------------
    # 2. 设置双边滤波参数
    # -------------------------------------------------
    # rho: 窗口半径 (Window Size)
    rho_val = 7

    # h: 颜色差异容忍度 (Color Sigma) -> 越小越能保留边缘，但去噪能力变弱
    h_val = 30

    # sigma_s: 空间距离容忍度 (Spatial Sigma) -> 对应公式里的 rho
    # 也就是离多远就算“远”了。通常 sigma_s 设为 rho 的一半左右比较合适
    sigma_s_val = 4.0

    # 3. 运行双边滤波
    denoised_img = bilateral_filter_custom(noisy_img_float, rho=rho_val, h=h_val, sigma_s=sigma_s_val)

    # 4. 定义输出路径
    output_filename = f'../../../out/baseline/neighborhood_filter/lena_gray_bilateral_h{h_val}_rho{rho_val}_s{int(sigma_s_val)}.png'

    # 5. 保存结果
    cv2.imwrite(output_filename, denoised_img)

    # 6. 计算指标 (Noise vs Original)
    print("-" * 30)
    noise_metrics = getMetrics.calculate_metrics(original.astype(np.uint8), noisy_img.astype(np.uint8))
    print(f"【原带噪图】 PSNR: {noise_metrics['PSNR']:.2f} | SSIM: {noise_metrics['SSIM']:.4f}")

    # 7. 计算指标 (Denoised vs Original)
    denoised_metrics = getMetrics.calculate_metrics(original.astype(np.uint8), denoised_img)
    print(f"【双边滤波】 PSNR: {denoised_metrics['PSNR']:.2f} | SSIM: {denoised_metrics['SSIM']:.4f}")
    print("-" * 30)
    print(f"处理完成！图片已保存为: {output_filename}")

    lpips, _ = AI_Metrics.compare_advanced_metrics(original_path, output_filename)
    print(f"{lpips:.4f} ")