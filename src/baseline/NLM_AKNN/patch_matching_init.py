import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import njit, prange
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numba import njit, prange
import time

def compute_patch_distance(img, y, x, ny, nx, patch_size):
    """
    辅助函数：计算两个 Patch 之间的距离 (Sum of Squared Differences)。
    这里为了演示逻辑简单实现，实际应用中可以使用积分图或卷积加速。

    参数:
    img: 图像数组 (H, W, C)
    y, x: 源像素坐标
    ny, nx: 目标(邻居)像素坐标
    patch_size: Patch 的边长 (例如 7)
    """
    h, w= img.shape
    r = patch_size // 2

    # 确定 Patch 的范围，注意处理图像边界
    y_min, y_max = max(0, y - r), min(h, y + r + 1)
    x_min, x_max = max(0, x - r), min(w, x + r + 1)

    # 对应的邻居 Patch 范围
    # 注意：如果源 Patch 在边界被截断，目标 Patch 也应取相同大小的区域以进行比较
    # 这里简化处理：只计算有效重叠区域，或假设 Patch 是完整的。
    # 为了代码健壮性，这里取对应偏移后的切片：

    patch_src = img[y_min:y_max, x_min:x_max]

    # 计算目标区域的起始点
    ny_min = ny - (y - y_min)
    nx_min = nx - (x - x_min)
    ny_max = ny_min + patch_src.shape[0]
    nx_max = nx_min + patch_src.shape[1]

    # 检查目标区域是否越界
    if ny_min < 0 or nx_min < 0 or ny_max > h or nx_max > w:
        return float('inf')  # 越界视为无穷大距离

    patch_target = img[ny_min:ny_max, nx_min:nx_max]

    # 计算 SSD (Sum of Squared Differences)
    diff = patch_src - patch_target
    dist = np.sum(diff * diff)
    return dist


def initialize_aknn(img, K, patch_size=7):
    """
    根据论文描述实现初始化过程。

    参数:
    img: 输入图像，numpy array, shape (H, W, C)
    K: 近邻数量 (K-nearest neighbors)
    patch_size: Patch 的大小

    返回:
    nn_offsets: 初始化后的偏移量场，shape (H, W, K, 2) -> 最后一维存储 (dy, dx)
    nn_dists: 对应的距离场，shape (H, W, K)
    """
    H, W = img.shape

    # 1. 参数设置: sigma_s = w / 3
    sigma_s = W / 3.0

    # 2. 生成随机偏移量 vi = sigma_s * ni (Eqn. 3)
    # ni 是标准正态分布
    # shape: (H, W, K, 2)
    ni = np.random.randn(H, W, K, 2)

    # 应用公式
    vi = sigma_s * ni

    # 偏移量必须是整数（像素坐标）
    vi = np.round(vi).astype(int)

    print(vi)
    # 初始化输出容器
    # nn_offsets 存储 K 个最好的偏移量 (y, x)
    nn_offsets = np.zeros((H, W, K, 2), dtype=int)
    # nn_dists 存储对应的距离，初始化为无穷大
    nn_dists = np.full((H, W, K), float('inf'))

    print(f"Initializing AKNN for image {H}x{W} with K={K}...")

    # 3. 填充优先队列 (此处通过排序模拟优先队列)
    # 由于 Python 循环遍历像素太慢，这里展示逻辑。
    # 在实际的高性能 Python 实现中，通常会向量化操作。

    # 为了演示清晰，我们遍历每个像素进行初始化
    # 注意：这步比较耗时，实际工程中通常使用 Numba 或 Cython 加速
    for y in range(H):
        for x in range(W):
            candidates = []

            for k in range(K):
                # 获取随机生成的偏移量
                dy, dx = vi[y, x, k]

                # 计算目标坐标
                ny, nx = y + dy, x + dx

                # 检查边界，如果出界，重新生成一个随机位置（简单的策略）
                # 或者直接忽略（距离设为 inf）
                if 0 <= ny < H and 0 <= nx < W:
                    dist = compute_patch_distance(img, y, x, ny, nx, patch_size)
                else:
                    dist = float('inf')
                    # 也可以选择此时随机选一个合法的点代替，保证队列不为空
                    ny, nx = np.random.randint(0, H), np.random.randint(0, W)
                    dy, dx = ny - y, nx - x
                    dist = compute_patch_distance(img, y, x, ny, nx, patch_size)

                candidates.append((dist, dy, dx))

            # 4. 维护顺序 (Priority Queue order)
            # 对 K 个候选者按距离排序 (升序)
            candidates.sort(key=lambda x: x[0])

            # 存入结果矩阵
            for k in range(K):
                nn_dists[y, x, k] = candidates[k][0]
                nn_offsets[y, x, k, 0] = candidates[k][1]  # dy
                nn_offsets[y, x, k, 1] = candidates[k][2]  # dx

    return nn_offsets, nn_dists






if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "data"
    noisy_path = DATA_DIR / "classic_photo_AWGN_sigma20_seed123456" / "lena_gray_sigma20_seed123456.png"
    noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
    if noisy_img is None:
        print(f"错误：找不到路径为 {noisy_path} 的图片，请检查路径。")
    img_float = noisy_img.astype(np.float32) / 255.0
    K = 5  # 我们想找 5 个最近邻
    patch_size = 7  # 补丁大小

    # 调用初始化函数
    # 返回的 offsets 形状: (H, W, K, 2)
    # 返回的 dists 形状: (H, W, K)
    offsets, dists = initialize_aknn(img_float, K, patch_size)
    print("\n初始化完成！")
    print(f"Offset Map Shape: {offsets.shape}")
    print(f"Distance Map Shape: {dists.shape}")

    # 可视化：看看每个像素目前找到的“第1名”匹配的距离是多少
    # 在随机初始化阶段，这个图看起来应该是一片杂乱的噪声（因为是随机猜的）
    best_dists = dists[:, :, 0]  # 取出 K=0 (最好的那个) 的距离图
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import njit, prange
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numba import njit, prange
import time

def compute_patch_distance(img, y, x, ny, nx, patch_size):
    """
    辅助函数：计算两个 Patch 之间的距离 (Sum of Squared Differences)。
    这里为了演示逻辑简单实现，实际应用中可以使用积分图或卷积加速。

    参数:
    img: 图像数组 (H, W, C)
    y, x: 源像素坐标
    ny, nx: 目标(邻居)像素坐标
    patch_size: Patch 的边长 (例如 7)
    """
    h, w= img.shape
    r = patch_size // 2

    # 确定 Patch 的范围，注意处理图像边界
    y_min, y_max = max(0, y - r), min(h, y + r + 1)
    x_min, x_max = max(0, x - r), min(w, x + r + 1)

    # 对应的邻居 Patch 范围
    # 注意：如果源 Patch 在边界被截断，目标 Patch 也应取相同大小的区域以进行比较
    # 这里简化处理：只计算有效重叠区域，或假设 Patch 是完整的。
    # 为了代码健壮性，这里取对应偏移后的切片：

    patch_src = img[y_min:y_max, x_min:x_max]

    # 计算目标区域的起始点
    ny_min = ny - (y - y_min)
    nx_min = nx - (x - x_min)
    ny_max = ny_min + patch_src.shape[0]
    nx_max = nx_min + patch_src.shape[1]

    # 检查目标区域是否越界
    if ny_min < 0 or nx_min < 0 or ny_max > h or nx_max > w:
        return float('inf')  # 越界视为无穷大距离

    patch_target = img[ny_min:ny_max, nx_min:nx_max]

    # 计算 SSD (Sum of Squared Differences)
    diff = patch_src - patch_target
    dist = np.sum(diff * diff)
    return dist


def initialize_aknn(img, K, patch_size=7):
    """
    根据论文描述实现初始化过程。

    参数:
    img: 输入图像，numpy array, shape (H, W, C)
    K: 近邻数量 (K-nearest neighbors)
    patch_size: Patch 的大小

    返回:
    nn_offsets: 初始化后的偏移量场，shape (H, W, K, 2) -> 最后一维存储 (dy, dx)
    nn_dists: 对应的距离场，shape (H, W, K)
    """
    H, W = img.shape

    # 1. 参数设置: sigma_s = w / 3
    sigma_s = W / 3.0

    # 2. 生成随机偏移量 vi = sigma_s * ni (Eqn. 3)
    # ni 是标准正态分布
    # shape: (H, W, K, 2)
    ni = np.random.randn(H, W, K, 2)

    # 应用公式
    vi = sigma_s * ni

    # 偏移量必须是整数（像素坐标）
    vi = np.round(vi).astype(int)

    print(vi)
    # 初始化输出容器
    # nn_offsets 存储 K 个最好的偏移量 (y, x)
    nn_offsets = np.zeros((H, W, K, 2), dtype=int)
    # nn_dists 存储对应的距离，初始化为无穷大
    nn_dists = np.full((H, W, K), float('inf'))

    print(f"Initializing AKNN for image {H}x{W} with K={K}...")

    # 3. 填充优先队列 (此处通过排序模拟优先队列)
    # 由于 Python 循环遍历像素太慢，这里展示逻辑。
    # 在实际的高性能 Python 实现中，通常会向量化操作。

    # 为了演示清晰，我们遍历每个像素进行初始化
    # 注意：这步比较耗时，实际工程中通常使用 Numba 或 Cython 加速
    for y in tqdm(range(H), desc="    Random", leave=False):
        for x in range(W):
            candidates = []

            for k in range(K):
                # 获取随机生成的偏移量
                dy, dx = vi[y, x, k]

                # 计算目标坐标
                ny, nx = y + dy, x + dx

                # 检查边界，如果出界，重新生成一个随机位置（简单的策略）
                # 或者直接忽略（距离设为 inf）
                if 0 <= ny < H and 0 <= nx < W:
                    dist = compute_patch_distance(img, y, x, ny, nx, patch_size)
                else:
                    dist = float('inf')
                    # 也可以选择此时随机选一个合法的点代替，保证队列不为空
                    ny, nx = np.random.randint(0, H), np.random.randint(0, W)
                    dy, dx = ny - y, nx - x
                    dist = compute_patch_distance(img, y, x, ny, nx, patch_size)

                candidates.append((dist, dy, dx))

            # 4. 维护顺序 (Priority Queue order)
            # 对 K 个候选者按距离排序 (升序)
            candidates.sort(key=lambda x: x[0])

            # 存入结果矩阵
            for k in range(K):
                nn_dists[y, x, k] = candidates[k][0]
                nn_offsets[y, x, k, 0] = candidates[k][1]  # dy
                nn_offsets[y, x, k, 1] = candidates[k][2]  # dx

    return nn_offsets, nn_dists




def visualize_pixel_and_candidates(img, y0, x0, offsets, patch_size):
    """
    可视化某个像素的 K 个候选
    红框：中心 patch
    蓝框：候选 patch
    """
    H, W = img.shape
    K = offsets.shape[2]
    r = patch_size // 2

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Pixel ({y0}, {x0}) and its {K} candidates")
    ax.axis('off')

    # 画中心像素 patch（红色）
    red_rect = patches.Rectangle(
        (x0 - r, y0 - r), patch_size, patch_size,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(red_rect)

    # 画候选 patch（蓝色）
    for k in range(K):
        dy, dx = offsets[y0, x0, k]
        ny, nx = y0 + dy, x0 + dx

        if 0 <= ny < H and 0 <= nx < W:
            blue_rect = patches.Rectangle(
                (nx - r, ny - r), patch_size, patch_size,
                linewidth=1.5, edgecolor='blue', facecolor='none'
            )
            ax.add_patch(blue_rect)

            # 标号（可选）
            ax.text(nx, ny, f"{k}", color='blue', fontsize=10)

    plt.show()


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "data"
    noisy_path = DATA_DIR / "classic_photo_AWGN_sigma20_seed123456" / "lena_gray_sigma20_seed123456.png"
    noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
    if noisy_img is None:
        print(f"错误：找不到路径为 {noisy_path} 的图片，请检查路径。")
    img_float = noisy_img.astype(np.float32) / 255.0
    K = 5  # 我们想找 5 个最近邻
    patch_size = 7  # 补丁大小

    # 调用初始化函数
    # 返回的 offsets 形状: (H, W, K, 2)
    # 返回的 dists 形状: (H, W, K)
    offsets, dists = initialize_aknn(img_float, K, patch_size)
    print("\n初始化完成！")
    print(f"Offset Map Shape: {offsets.shape}")
    print(f"Distance Map Shape: {dists.shape}")
    y0, x0 = 16, 16

    visualize_pixel_and_candidates(
        img_float,
        y0,
        x0,
        offsets,
        patch_size
    )
