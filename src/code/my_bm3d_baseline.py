import numpy as np
import cv2
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


# -------------------------
# 1. 核心搜索函数 (SAD 距离)
# -------------------------
def get_sad(p1, p2):
    return np.sum(np.abs(p1 - p2))


def update_best_k(img, y, x, prop_dy, prop_dx, offsets, dists, patch_size, K):
    H, W = img.shape
    ny, nx = y + prop_dy, x + prop_dx
    r = patch_size // 2

    # 边界检查
    if ny - r < 0 or ny + r >= H or nx - r < 0 or nx + r >= W:
        return

    # 提取 Patch 并计算 SAD
    p_ref = img[y - r: y + r + 1, x - r: x + r + 1]
    p_cand = img[ny - r: ny + r + 1, nx - r: nx + r + 1]
    dist = np.sum(np.abs(p_ref - p_cand))

    # 如果比当前最差的还好，且不是同一个位置，则插入
    if dist < dists[y, x, -1]:
        # 查重
        for k in range(K):
            if offsets[y, x, k, 0] == prop_dy and offsets[y, x, k, 1] == prop_dx:
                return

        # 插入并排序
        dists[y, x, -1] = dist
        offsets[y, x, -1] = [prop_dy, prop_dx]

        # 简单排序保持 dists 有序
        idx = np.argsort(dists[y, x])
        dists[y, x] = dists[y, x, idx]
        offsets[y, x] = offsets[y, x, idx]


# -------------------------
# 2. 降噪函数 (协同平均)
# -------------------------
def collaborative_denoise(noisy_img, final_offsets):
    H, W = noisy_img.shape
    K = final_offsets.shape[2]
    denoised_img = np.zeros_like(noisy_img)

    print("正在进行协同平均降噪...")
    for y in range(H):
        for x in range(W):
            pixel_values = []
            for k in range(K):
                dy, dx = final_offsets[y, x, k]
                ny, nx = y + dy, x + dx
                pixel_values.append(noisy_img[ny, nx])

            # 取 K 个相似像素的平均值
            denoised_img[y, x] = np.mean(pixel_values)

    return denoised_img


# -------------------------
# 3. 主实验流程
# -------------------------
def run_denoise_test():
    # A. 读取并裁剪图片 (为了不加速也能跑，先切一小块)
    raw_img = cv2.imread("data/classic_photo/lena_gray.png", cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        print("未找到图片，请检查路径")
        return

    # 裁剪 128x128 区域演示
    img = raw_img[200:328, 200:328].astype(np.float32) / 255.0
    H, W = img.shape

    # B. 添加噪声 (Sigma = 20)
    sigma = 20 / 255.0
    noisy_img = img + np.random.normal(0, sigma, img.shape)
    noisy_img = np.clip(noisy_img, 0, 1)

    # C. 初始化 AKNN
    K = 8
    patch_size = 7
    offsets = np.random.randint(-15, 16, (H, W, K, 2))
    dists = np.full((H, W, K), 1e6)

    # D. 开始 AKNN 搜索迭代
    print(f"开始 AKNN 搜索 (4次迭代)...")
    for i in range(4):
        # 简单的传播与随机搜索合并演示
        for y in tqdm(range(H), desc=f"第 {i + 1} 轮扫描"):
            for x in range(W):
                # 1. 检查邻居 (传播)
                if y > 0: update_best_k(noisy_img, y, x, offsets[y - 1, x, 0, 0], offsets[y - 1, x, 0, 1], offsets,
                                        dists, patch_size, K)
                if x > 0: update_best_k(noisy_img, y, x, offsets[y, x - 1, 0, 0], offsets[y, x - 1, 0, 1], offsets,
                                        dists, patch_size, K)

                # 2. 随机搜索 (加窗)
                rad = max(1, int((W // 4) * (0.5 ** i)))
                ry, rx = np.random.randint(-rad, rad + 1), np.random.randint(-rad, rad + 1)
                update_best_k(noisy_img, y, x, offsets[y, x, 0, 0] + ry, offsets[y, x, 0, 1] + rx, offsets, dists,
                              patch_size, K)

    # E. 执行降噪
    denoised_result = collaborative_denoise(noisy_img, offsets)

    # F. 结果对比
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1);
    plt.title("Original (Clean)");
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 3, 2);
    plt.title("Noisy (Input)");
    plt.imshow(noisy_img, cmap='gray')
    plt.subplot(1, 3, 3);
    plt.title(f"Denoised (K={K} Mean)");
    plt.imshow(denoised_result, cmap='gray')
    plt.show()


if __name__ == "__main__":
    run_denoise_test()