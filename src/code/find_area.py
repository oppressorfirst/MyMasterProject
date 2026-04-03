import cv2
import numpy as np
import time
import concurrent.futures
import csv
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# =====================================================================
# 1. 核心导入：从你的 v6 版本中导入所有必需的底层处理函数
# =====================================================================
# 确保 AKNN_BM3D_v6.py 在同级目录，并且去掉了 v6 里面最下方自动运行的测试代码
# (或者 v6 里的测试代码已经被 if __name__ == "__main__": 保护起来了)
from AKNN_BM3D_v6 import (
    read_png_to_yuv,
    add_poisson_gaussian_noise,
    forward_gat,
    inverse_gat,
    process_single_block
)


# =====================================================================
# 2. 动态网格切分函数 (专为搜索范围探索设计)
# =====================================================================
def split_image_into_grid(img, block_size, overlap=16):
    """
    将图像动态切分为指定大小的网格块，以精确限制 AKNN 的物理搜索范围。
    参数:
        img: 输入图像 (H, W)
        block_size: 搜索范围/块大小 (如 39, 64, 128, 256)
        overlap: 块与块之间的重叠像素数 (用于消除边界伪影)
    返回:
        blocks: 图像块数据列表
        coords: 对应的坐标 [(y0, y1, x0, x1), ...]
    """
    H, W = img.shape[:2]
    coords = []
    blocks = []
    
    # 计算滑动的步长
    stride = block_size - overlap
    if stride <= 0:
        raise ValueError("Overlap 必须小于 block_size！")

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y0, x0 = y, x
            y1 = min(H, y + block_size)
            x1 = min(W, x + block_size)
            
            # 如果到达边缘，强行向回对齐，保证传入算法的块大小一致
            if y1 - y0 < block_size and H >= block_size:
                y0 = H - block_size
                y1 = H
            if x1 - x0 < block_size and W >= block_size:
                x0 = W - block_size
                x1 = W

            # 避免重复添加完全相同的坐标（在极端边缘处可能发生）
            if (y0, y1, x0, x1) not in coords:
                coords.append((y0, y1, x0, x1))
                blocks.append(img[y0:y1, x0:x1].copy())

    return blocks, coords


# =====================================================================
# 3. 自动化架构探索 (Design Space Exploration) 主程序
# =====================================================================
if __name__ == "__main__":
    # --- 测试参数配置 ---
    dataset_path = "data/PhotoCD_PCD0992"
    csv_file_path = Path(dataset_path) / "AKNN_Search_Range_Exploration.csv"
    
    # 核心探索空间：我们要测试的搜索范围 (Tile Size) 列表
    search_ranges = [64, 96, 128, 192, 256]
    overlap_pixels = 39  # 重叠区 16 个像素足矣，过大增加冗余计算
    
    # 算法基础参数 (对齐你原本的设定)
    sigma_val = 25
    sigma_norm = sigma_val / 255.0
    a_val = 0.02
    K = 7
    patch_size = 7
    process_step = 2
    
    # 为了快速拿到扫参结果，选取视频的前 3 帧作为评估基准
    test_frames = range(1,25) 

    print(f"准备开始测试，结果将写入: {csv_file_path}")

    # 打开 CSV 文件准备写入
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # 写入包含 Search_Range 的新表头
        writer.writerow(['Search_Range', 'Image_Index', 'Total_Blocks', 'Time_Seconds', 'PSNR_dB', 'SSIM'])

        # 外层循环：遍历所有的搜索范围
        for s_range in search_ranges:
            print(f"\n" + "="*50)
            print(f"🚀 开始测试搜索范围 (SRAM Tile Size): {s_range} x {s_range}")
            print(f"="*50)

            # 内层循环：遍历测试帧
            for idx in test_frames:
                clean_path = Path(dataset_path) / f"{idx:02d}.png"
                y, cb, cr, clean_img_cv = read_png_to_yuv(str(clean_path))

                if clean_img_cv is None:
                    print(f"找不到图片: {clean_path}")
                    continue

                np.random.seed(42)
                y_noisy = add_poisson_gaussian_noise(y, a=a_val, sigma_norm=sigma_norm, seed=42)
                
                # VST 与高斯向导图生成
                y_noisy_vst = forward_gat(y_noisy, a=a_val, sigma=sigma_norm)
                guide_img_vst = cv2.GaussianBlur(y_noisy_vst, (5, 5), 1.5)

                # 核心：根据当前的 s_range 进行动态网格切分
                noisy_blocks, block_coords = split_image_into_grid(y_noisy_vst, block_size=s_range, overlap=overlap_pixels)
                guide_blocks, _ = split_image_into_grid(guide_img_vst, block_size=s_range, overlap=overlap_pixels)
                
                num_blocks = len(block_coords)
                print(f"  > 帧 {idx:02d}: 被切分为 {num_blocks} 个 {s_range}x{s_range} 的宏块")

                denoised_vst_blocks = [None] * num_blocks
                t_start_parallel = time.time()

                # 并行处理所有的块 (利用你 v6 里的多核加速)
                # 注意：如果块数太多(如39x39时)，可以适当调大 max_workers
                with concurrent.futures.ProcessPoolExecutor(max_workers=40) as executor: 
                    futures = []
                    for i in range(num_blocks):
                        future = executor.submit(
                            process_single_block,
                            i, noisy_blocks[i], guide_blocks[i],
                            K, patch_size, process_step
                        )
                        futures.append(future)

                    for future in concurrent.futures.as_completed(futures):
                        block_idx, result_block = future.result()
                        denoised_vst_blocks[block_idx] = result_block

                duration = time.time() - t_start_parallel
                print(f"  > 并行降噪完成，耗时: {duration:.2f} 秒")

                # 图像无缝重组融合
                H, W = y.shape
                numerator = np.zeros((H, W), dtype=np.float32)
                denominator = np.zeros((H, W), dtype=np.float32)

                for i in range(num_blocks):
                    y0, y1, x0, x1 = block_coords[i]
                    numerator[y0:y1, x0:x1] += denoised_vst_blocks[i]
                    denominator[y0:y1, x0:x1] += 1.0 

                y_denoised_vst = numerator / denominator
                
                # 逆 VST 变换
                y_denoised = inverse_gat(y_denoised_vst, a=a_val, sigma=sigma_norm)
                y_denoised = np.clip(y_denoised, 0.0, 1.0)

                # 计算指标
                current_psnr = psnr(y, y_denoised, data_range=1.0)
                current_ssim = ssim(y, y_denoised, data_range=1.0)

                print(f"  > 结果: PSNR = {current_psnr:.2f} dB | SSIM = {current_ssim:.4f}")

                # 记录当前测试点到 CSV 并刷新
                writer.writerow([s_range, idx, num_blocks, round(duration, 2), round(current_psnr, 2), round(current_ssim, 4)])
                csv_file.flush()

    print(f"\n✅ 所有架构探索测试已完成！请查看 {csv_file_path} 绘制效能曲线。")
