import numpy as np
import cv2

# 添加 utils 文件夹到 Python 路径
from utils import getMetrics, AI_Metrics
import time

def yaroslavsky_neighborhood_filter(image, rho, h):
    """
    实现了图片中的 Yaroslavsky Neighborhood Filter 公式。

    参数:
    - image: 输入的灰度图像 (2D numpy array)
    - rho: 邻域半径 (对应公式中的 B_rho(x))
    - h: 滤波参数 (对应公式中的 h，控制对差异的敏感度)

    返回:
    - 滤波后的图像
    """
    # 1. 获取图像尺寸
    height, width = image.shape

    # 2. 转换成浮点数进行计算，防止溢出
    img_float = image.astype(np.float64)
    output = np.zeros_like(img_float)

    # 3. 对图像进行边缘填充 (Padding)
    # 这样边缘的像素也能有完整的邻域窗口
    padded_img = cv2.copyMakeBorder(img_float, rho, rho, rho, rho, cv2.BORDER_REFLECT)

    # h的平方，避免在循环里重复计算
    h2 = h ** 2

    print(f"开始处理图像 (尺寸: {width}x{height})... 这可能需要一点时间")

    # 4. 遍历每一个像素 (对应公式里的 x)
    for y in range(height):
        for x in range(width):
            # --- 这一段完全对应你的公式 ---

            # 这里的 center_pixel 就是 u(x)
            center_val = padded_img[y + rho, x + rho]

            # 提取邻域 B_rho(x) (即周围的小方块)
            # 这里的 window 就是 u(y) 的集合
            window = padded_img[y:y + 2 * rho + 1, x:x + 2 * rho + 1]

            # 计算强度差的平方: |u(y) - u(x)|^2
            diff_sq = (window - center_val) ** 2

            # 计算权重: exp(- diff / h^2)
            weights = np.exp(-diff_sq / h2)

            # 计算归一化系数 C(x) (对应公式分母)
            normalization_factor = np.sum(weights)

            # 计算加权和 (对应公式分子中的积分)
            weighted_sum = np.sum(window * weights)

            # 最终结果 = 加权和 / 归一化系数
            output[y, x] = weighted_sum / normalization_factor

            # ---------------------------

    # 将结果转回 0-255 的整数格式
    return np.clip(output, 0, 255).astype(np.uint8)


# ==========================================
# 主程序：生成噪点图并测试
# ==========================================

# 1. 你可以读取你自己的图
original_path = "../../../data/classic_photo/lena_gray.png"
original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
original = original.astype(np.float32)

noisy_img = cv2.imread("../../../data/classic_photo_AWGN_sigma20_seed123456/lena_gray_sigma20_seed123456.png", cv2.IMREAD_GRAYSCALE)
noisy_img = noisy_img.astype(np.float32)

# 3. 设置参数 (你可以调整这两个数看看效果！)
# rho = 3: 表示看周围半径为3的区域 (7x7的窗口)
# h = 30: 表示如果两个像素差值超过30，权重就会变得很小（不参考它）
rho_val = 3
h_val = 30

# 4. 运行我们写的滤波器
start_time = time.perf_counter()
denoised_img = yaroslavsky_neighborhood_filter(noisy_img, rho=rho_val, h=h_val)
end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"处理耗时: {elapsed_time:.4f} 秒")
output_filename = f'../../../out/images/baseline/neighborhood_filter/lena_gray_yaroslavsky_h{h_val}_rho{rho_val}.png'





# 5. 显示结果
cv2.imwrite(output_filename, denoised_img)

noise_metrics = getMetrics.calculate_metrics(original.astype(np.uint8), noisy_img.astype(np.uint8))
print(f"噪声PSNR: {noise_metrics['PSNR']:.2f} dB")
print(f"噪声SSIM: {noise_metrics['SSIM']:.4f}")
print(f"噪声MSE: {noise_metrics['MSE']:.2f}")
# 6. 计算评价指标

denoised_metrics = getMetrics.calculate_metrics(original.astype(np.uint8), denoised_img)
print(f"处理完成！图片已保存为: {output_filename}")
print(f"去噪后PSNR: {denoised_metrics['PSNR']:.2f} dB")
print(f"去噪后SSIM: {denoised_metrics['SSIM']:.4f}")
print(f"去噪后MSE: {denoised_metrics['MSE']:.2f}")

lpips, _ = AI_Metrics.compare_advanced_metrics(original_path, output_filename)
print(f"{lpips:.4f} ")