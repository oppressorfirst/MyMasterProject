import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import njit, prange
from tqdm import tqdm, trange
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numba import njit, prange
import time

from AKNN_init import initialize_aknn,visualize_pixel_and_candidates

# --- 1. 核心辅助函数：维护优先队列 ---
def update_best_k(img, y, x, prop_dy, prop_dx, offsets, dists, patch_size, H, W, K):
    """
    尝试将一个新的偏移量 (prop_dy, prop_dx) 插入到像素 (y, x) 的 K 个最佳列表中。
    逻辑：
    1. 计算应用该偏移量后的距离。
    2. 如果距离比当前列表中最差的一个要小，插入并保持列表有序。
    """

    # 计算目标位置
    ny, nx = y + prop_dy, x + prop_dx

    # 1. 越界检查 (Target Patch 不能跑出图外)
    r = patch_size // 2
    if y - r < 0 or y + r >= H or x - r < 0 or x + r >= W:
        return
    if ny - r < 0 or ny + r >= H or nx - r < 0 or nx + r >= W:
        return  # 越界无效，直接跳过

    # 2. 计算距离 (SSD)
    # 提取当前 Patch 和 目标 Patch
    # 注意：这里切片操作在纯 Python 中比循环快，所以保留切片
    patch_src = img[y - r: y + r + 1, x - r: x + r + 1]
    patch_tgt = img[ny - r: ny + r + 1, nx - r: nx + r + 1]
    if patch_src.shape != (patch_size, patch_size) or patch_tgt.shape != (patch_size, patch_size):
        return
    diff = patch_src - patch_tgt
    new_dist = np.sum(diff * diff)

    # 3. 检查是否值得插入
    # dists[y, x] 已经是升序排列的。
    # 如果新距离 >= 当前第 K 个（最差的）距离，说明这个新偏移量不够好，直接扔掉
    current_dists = dists[y, x]
    if new_dist >= current_dists[-1]:
        return

    # 4. 插入逻辑 (模拟优先队列)
    # 我们需要把新结果插入到正确的位置，保持列表有序

    # 获取当前像素的偏移量列表
    current_offsets = offsets[y, x]

    # 简单查重：如果这个偏移量已经在列表里了，就不要重复添加
    for k in range(K):
        if current_offsets[k][0] == prop_dy and current_offsets[k][1] == prop_dx:
            return

    # 找到插入位置
    insert_pos = -1
    for k in range(K):
        if new_dist < current_dists[k]:
            insert_pos = k
            break

    # 执行插入 (移动数组元素)
    if insert_pos != -1:
        # 比如列表是 [10, 20, 30, 40, 50], 新来个 15
        # insert_pos = 1
        # 我们把 15 插在索引 1 的位置，后面的后移，把 50 挤出去

        # 倒序移动，把位置腾出来
        for k in range(K - 1, insert_pos, -1):
            current_dists[k] = current_dists[k - 1]
            current_offsets[k] = current_offsets[k - 1]

        # 填入新值
        current_dists[insert_pos] = new_dist
        current_offsets[insert_pos] = [prop_dy, prop_dx]


# --- 2. 传播步骤 (Propagation) ---
def propagation_step(img, offsets, dists, patch_size, iter_num):
    H, W = img.shape[:2]
    K = offsets.shape[2]

    print(f"  > Propagation (Direction: {'Scanline' if iter_num % 2 == 0 else 'Reverse'})...")

    # 根据迭代次数决定扫描方向
    if iter_num % 2 == 0:
        # 正向扫描: 左上 -> 右下
        # 我们从 (1, 1) 开始，这样 (y-1) 和 (x-1) 肯定存在，不用频繁检查边界
        y_range = range(1, H)
        x_range = range(1, W)
        neighbor_deltas = [(-1, 0), (0, -1)]  # 邻居：上，左
    else:
        # 反向扫描: 右下 -> 左上
        # 从 H-2, W-2 开始，倒着走
        y_range = range(H - 2, -1, -1)
        x_range = range(W - 2, -1, -1)
        neighbor_deltas = [(1, 0), (0, 1)]  # 邻居：下，右

    # 遍历每个像素
    for y in tqdm(y_range, desc=f"    Prop iter{iter_num+1}", leave=False):
        for x in x_range:
            # 检查它的两个邻居
            for dy_n, dx_n in neighbor_deltas:
                # 邻居的坐标
                nb_y, nb_x = y + dy_n, x + dx_n

                # 获取邻居目前认为最好的 K 个偏移量
                # "既然你是我的邻居，那你的匹配偏移量，可能也适合我"
                nb_offsets = offsets[nb_y, nb_x]  # shape (K, 2)

                for k in range(K):
                    prop_dy, prop_dx = nb_offsets[k]

                    # 尝试用邻居的偏移量来更新当前像素 (y, x)
                    update_best_k(img, y, x, prop_dy, prop_dx,
                                  offsets, dists, patch_size, H, W, K)


# --- 3. 随机搜索步骤 (Random Search) ---
def random_search_step(img, offsets, dists, patch_size, search_radius):
    H, W = img.shape[:2]
    K = offsets.shape[2]

    print(f"  > Random Search (Radius: {search_radius:.2f})...")

    # 遍历每个像素
    for y in tqdm(range(H), desc="    Random", leave=False):
        for x in range(W):
            # 对当前已经找到的 K 个最佳偏移量，每一个都试着在其周围“抖动”一下
            for k in range(K):
                best_dy, best_dx = offsets[y, x, k]

                # 生成随机扰动
                # 使用标准正态分布 * 半径
                rand_dy = int(round(search_radius * np.random.randn()))
                rand_dx = int(round(search_radius * np.random.randn()))

                # 新的尝试偏移量 = 原来的最佳 + 随机扰动
                search_dy = best_dy + rand_dy
                search_dx = best_dx + rand_dx

                # 尝试更新
                update_best_k(img, y, x, search_dy, search_dx,
                              offsets, dists, patch_size, H, W, K)


# --- 4. 主程序：把所有步骤串起来 ---
def run_aknn_pure_python(img, init_offsets, init_dists, iterations, patch_size):
    """
    AKNN 主循环：不使用 Numba，纯 Python 逻辑演示
    """
    H, W = img.shape[:2]
    K = init_offsets.shape[2]

    # 复制一份，以免修改原始输入
    offsets = init_offsets.copy()
    dists = init_dists.copy()

    # 初始搜索半径 (根据论文: width / 3)
    # 但为了演示和收敛速度，这里设置起始半径
    search_radius = W / 3.0

    print(f"Starting AKNN Loop ({iterations} iterations)...")

    for i in trange(iterations, desc="AKNN Iter"):
        t0 = time.time()

        # A. 传播
        propagation_step(img, offsets, dists, patch_size, i)

        # B. 随机搜索
        # 半径每次减半 (alpha = 0.5)
        current_radius = search_radius * (0.5 ** i)
        if current_radius < 1: current_radius = 1

        random_search_step(img, offsets, dists, patch_size, current_radius)

        t1 = time.time()
        tqdm.write(f"Iteration {i + 1} finished in {t1 - t0:.2f}s")

    return offsets, dists


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "data"
    noisy_path = DATA_DIR / "classic_photo_AWGN_sigma20_seed123456" / "lena_gray_sigma20_seed123456.png"
    noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
    if noisy_img is None:
        print(f"错误：找不到路径为 {noisy_path} 的图片，请检查路径。")
    img_float = noisy_img.astype(np.float32) / 255.0
    K = 5  # 我们想找 5 个最近邻
    patch_size = 7 # 补丁大小
    offsets, dists = initialize_aknn(img_float, K, patch_size)
    final_offsets, final_dists = run_aknn_pure_python(img_float, offsets, dists, 4,patch_size)
    visualize_pixel_and_candidates(
        img_float,
        32,
        32,
        final_offsets,
        patch_size
    )
    visualize_pixel_and_candidates(
        img_float,
        64,
        64,
        final_offsets,
        patch_size
    )