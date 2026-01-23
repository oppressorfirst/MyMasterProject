from pathlib import Path
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from src.utils import getMetrics, AI_Metrics
import time
from tqdm import tqdm
from numba import njit

class BM3D_Stage1_Params:
    """
    对应论文 Table I 的 Normal Profile (sigma <= 40 的情况)
    """

    def __init__(self, sigma, patch_size, stride, window_size, top_k):
        self.sigma = sigma

        # --- 块与窗口参数 ---
        self.patch_size = patch_size  # 块大小
        self.stride = stride  # 滑动步长
        self.window_size = window_size  # 搜索窗口大小
        self.top_k = top_k  # 最大堆叠块数

        # --- 阈值参数 ---
        # 3D 变换的硬阈值 (lambda_3d * sigma)
        # 论文中 lambda_3d = 2.7
        self.lambda_3d = 2.7

        # 块匹配阈值 (tau_match)
        # 论文 Table I: tau_match = 2500 (这是未归一化的 SSD 阈值，针对 255 范围)
        # 如果图像是 0-1 范围，需要换算：2500 / (255^2)
        # 或者我们在计算距离时把图像乘回 255
        self.tau_match = 2500


def get_kaiser_window(patch, beta=2.0):
    """
    生成 2D Kaiser 窗 (论文 IV. Reduction of border effects)
    用于聚合时的加权，消除块效应。
    """
    k = np.kaiser(patch, beta)
    win_2d = np.outer(k, k)
    # win_2d /= np.max(win_2d)
    return win_2d.astype(np.float32)


def read_gray_image(path):
    """
   读取进行降噪的图片
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"找不到图片：{path}")
    img_float = img.astype(np.float32) / 255.0
    return img_float


def window_block_matching(img, patch_size, stride, window_size, top_k):
    H, W = img.shape
    all_results = []
    search_radius = window_size // 2

    for ref_y in range(0, H - patch_size + 1, stride):
        for ref_x in range(0, W - patch_size + 1, stride):

            ref_patch = img[ref_y:ref_y + patch_size, ref_x:ref_x + patch_size]

            matches = []
            # ---------- ① 限制搜索窗口 ----------
            y_min = max(0, ref_y - search_radius)
            y_max = min(H - patch_size, ref_y + search_radius)

            x_min = max(0, ref_x - search_radius)
            x_max = min(W - patch_size, ref_x + search_radius)

            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    cand_patch = img[y:y + patch_size, x:x + patch_size]
                    diff = ref_patch - cand_patch
                    dist = np.mean(diff * diff)

                    matches.append((dist, y, x))

            matches.sort(key=lambda t: t[0])
            top_matches = matches[:top_k]

            all_results.append({
                "ref_pos": (ref_y, ref_x),
                "top_matches": top_matches
            })
    return all_results


def window_block_matching_single(img, ref_y, ref_x, params):
    H, W = img.shape
    patch_size = params.patch_size
    search_radius = params.window_size // 2

    ref_patch = img[ref_y:ref_y + patch_size, ref_x:ref_x + patch_size]

    matches = [(0.0, ref_y, ref_x)]  # ✅ 强制 self-match

    # ---------- ① 限制搜索窗口 ----------
    y_min = max(0, ref_y - search_radius)
    y_max = min(H - patch_size, ref_y + search_radius)
    x_min = max(0, ref_x - search_radius)
    x_max = min(W - patch_size, ref_x + search_radius)

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if y == ref_y and x == ref_x:
                continue
            cand_patch = img[y:y + patch_size, x:x + patch_size]
            diff = ref_patch - cand_patch
            dist = np.sum(diff * diff)
            if dist < params.tau_match:
                matches.append((dist, y, x))

    matches.sort(key=lambda x: x[0])
    return matches[:params.top_k]


@njit
def patch_mse(img, y1, x1, y2, x2, patch_size):
    s = 0.0
    for i in range(patch_size):
        for j in range(patch_size):
            d = img[y1 + i, x1 + j] - img[y2 + i, x2 + j]
            s += d * d
    return s / (patch_size * patch_size)


@njit
def numba_window_match_single(img, ref_y, ref_x, patch_size, top_k, search_radius):
    H, W = img.shape

    dists = np.full(top_k, 1e9, dtype=np.float32)
    ys = np.zeros(top_k, dtype=np.int32)
    xs = np.zeros(top_k, dtype=np.int32)

    y_min = max(0, ref_y - search_radius)
    y_max = min(H - patch_size, ref_y + search_radius)
    x_min = max(0, ref_x - search_radius)
    x_max = min(W - patch_size, ref_x + search_radius)

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if y == ref_y and x == ref_x:
                continue

            d = patch_mse(img, ref_y, ref_x, y, x, patch_size)

            if d < dists[top_k - 1]:
                pos = top_k - 1
                while pos > 0 and d < dists[pos - 1]:
                    dists[pos] = dists[pos - 1]
                    ys[pos] = ys[pos - 1]
                    xs[pos] = xs[pos - 1]
                    pos -= 1
                dists[pos] = d
                ys[pos] = y
                xs[pos] = x

    return dists, ys, xs


def get_3d_group(img_padded, matches, patch_size):
    """
    根据匹配结果，从图中抓取块，堆叠成 3D 组
    matches: [(dist, y, x), ...]
    返回: group (K, N, N)
    """
    K = len(matches)
    group = np.zeros((K, patch_size, patch_size), dtype=np.float32)

    for i, (_, y, x) in enumerate(matches):
        group[i, :, :] = img_padded[y: y + patch_size, x: x + patch_size]

    return group


def collaborative_filtering(group_noisy, sigma, lambda_3d):
    """
    Step 3: 3D 变换 + 硬阈值 + 3D 逆变换
    """
    # 1. [3D Transform] = 2D DCT + 1D Haar
    # 先对每张图做 2D DCT (axis=1, 2)
    # type=2, norm='ortho' 是标准的 DCT-II
    coefs_2d = fft.dctn(group_noisy, axes=(1, 2), type=2, norm='ortho')

    # 再对组维度做 1D Haar (axis=0)
    # 这里为了手写方便，可以用 DCT 代替 Haar，或者手写一个简单的 Haar
    # 论文 Normal Profile 其实用的是 Haar [cite: 649]。
    # 为了复现，这里先用 DCT (axis=0) 模拟 1D 变换，它比 Haar 能量集中性更强，效果往往更好。
    # 如果必须严格复现 Haar，代码会繁琐一些，先用 DCT 跑通流程。
    coefs_3d = fft.dct(coefs_2d, axis=0, type=2, norm='ortho')

    # 2. [Hard Thresholding]
    # 阈值 = lambda_3d * sigma
    thresh = lambda_3d * sigma

    # 找出保留的系数 (非零)
    mask = np.abs(coefs_3d) > thresh
    coefs_hard = coefs_3d * mask

    # 统计非零系数个数 (用于计算权重) [cite: 569]
    # N_hard = number of non-zero coefficients
    nonzero_count = np.sum(mask)

    # 3. [Inverse 3D Transform]
    # 逆 1D
    group_rec_2d = fft.idct(coefs_hard, axis=0, type=2, norm='ortho')
    # 逆 2D
    group_estimate = fft.idctn(group_rec_2d, axes=(1, 2), type=2, norm='ortho')

    return group_estimate, nonzero_count


def aggregation(group_est, matches, nonzero_count, numerator, denominator, kaiser_window, sigma):
    """
    Step 4: 将去噪后的组放回累加器
    """
    # 1. 计算权重 (论文 Eq. 10)
    if nonzero_count >= 1:
        # 注意：这里 sigma 是标准差。有些实现会忽略 sigma^2 因子因为它是常数，分子分母会约掉。
        # 但严格复现应该加上。
        weight = 1.0 / (nonzero_count * sigma * sigma)
    else:
        weight = 1.0  # 避免除以零

    # 2. 遍历组里的每一个块，累加回去
    for i, (_, y, x) in enumerate(matches):
        block_est = group_est[i, :, :]

        # 分子: 累加 (值 * 权重 * 窗)
        numerator[y: y + block_est.shape[0], x: x + block_est.shape[1]] += \
            block_est * kaiser_window * weight

        # 分母: 累加 (权重 * 窗)
        denominator[y: y + block_est.shape[0], x: x + block_est.shape[1]] += \
            kaiser_window * weight


def run_bm3d_stage1(img_noisy, sigma, patch_size, stride, window_size, top_k):
    # 1. 初始化
    img_pad, num, den, kaiser, params, pad_w = init_bm3d_stage1(img_noisy, sigma, patch_size, stride, window_size, top_k)

    H, W = img_pad.shape
    N = params.patch_size
    # 触发 JIT 编译（第一次慢，后面飞快）
    print("Warming up Numba JIT...")
    _ = numba_window_match_single(noisy_img_float, 0, 0,
                            patch_size, top_k, 2)
    print("开始 BM3D Stage 1 处理...")
    ys = list(range(0, H - N + 1, params.stride))
    xs = list(range(0, W - N + 1, params.stride))
    total = len(ys) * len(xs)
    t_match = 0.0
    t_group = 0.0
    t_filt = 0.0
    t_aggr = 0.0
    t0_all = time.perf_counter()
    pbar = tqdm(total=total, desc="BM3D Stage1", dynamic_ncols=True)
    # 2. 遍历图像 (Reference Blocks)
    # 按照 stride 遍历
    for y in ys:
        for x in xs:
            # --- A. 块匹配 (Grouping) ---
            # 使用你之前写的 window_block_matching (注意要加上 self 匹配)
            # 这里调用参考实现:
            # ref_patch = img_pad[y:y+N, x:x+N]
            # matches = find_similar_blocks(img_pad, ref_patch, y, x, params)

            # 暂时用伪代码表示你上一轮写的匹配逻辑:
            t0 = time.perf_counter()

            dists, ys, xs = numba_window_match_single(
                noisy_img_float,
                y,
                x,
                patch_size,
                top_k,
                params.window_size,
            )


            matches = [(float(dists[i]), int(ys[i]), int(xs[i])) for i in range(top_k)]
            # matches = window_block_matching_single(img_pad, y, x, params)
            t_match += (time.perf_counter() - t0)

            # --- B. 构建 3D 组 ---
            t0 = time.perf_counter()
            group_noisy = get_3d_group(img_pad, matches, N)
            t_group += (time.perf_counter() - t0)
            # --- C. 协同滤波 (Collaborative Filtering) ---
            t0 = time.perf_counter()
            group_est, n_nonzero = collaborative_filtering(group_noisy, params.sigma, params.lambda_3d)
            t_filt += (time.perf_counter() - t0)
            # --- D. 聚合 (Aggregation) ---
            t0 = time.perf_counter()
            aggregation(group_est, matches, n_nonzero, num, den, kaiser, params.sigma)
            t_aggr += (time.perf_counter() - t0)

            pbar.update(1)
            if (pbar.n % 500) == 0:
                print(
                    f"[debug] matches={len(matches)}, best_dist={matches[0][0]:.2f}, worst_in_top={matches[-1][0]:.2f}")

            # 每隔一段更新一次 postfix，避免太频繁导致开销
            if (pbar.n % 200) == 0 or pbar.n == total:
                elapsed = time.perf_counter() - t0_all
                pbar.set_postfix({
                    "elapsed_s": f"{elapsed:.1f}",
                    "match%": f"{(t_match / elapsed * 100):.1f}",
                    "filt%": f"{(t_filt / elapsed * 100):.1f}",
                })

    pbar.close()
    t_all = time.perf_counter() - t0_all
    print("\n[Stage1] done.")
    print(f"  Total: {t_all:.3f} s")
    print(f"  Match: {t_match:.3f} s  ({t_match / t_all * 100:.1f}%)")
    print(f"  Group: {t_group:.3f} s  ({t_group / t_all * 100:.1f}%)")
    print(f"  Filt : {t_filt:.3f} s  ({t_filt / t_all * 100:.1f}%)")
    print(f"  Aggr : {t_aggr:.3f} s  ({t_aggr / t_all * 100:.1f}%)")
    # 3. 最终除法 (Final Estimate)
    # 裁剪掉 padding
    est_pad = num / (den + 1e-12)  # 防止除零

    # Crop back to original size
    est_img = est_pad[pad_w:-pad_w, pad_w:-pad_w]

    return est_img


def init_bm3d_stage1(img_noisy, sigma_val ,patch_size, stride, window_size, top_k):
    """
    第一步：初始化环境
    """
    # 1. 实例化参数
    p = BM3D_Stage1_Params(sigma_val, patch_size, stride, window_size, top_k)

    # 2. 图像预处理 (Padding)
    # 搜索窗口是 39，半径大概是 19。为了防止边界越界，pad 一下。
    pad_w = p.window_size // 2
    img_pad = np.pad(img_noisy, ((pad_w, pad_w), (pad_w, pad_w)), mode='reflect')

    # 3. 初始化聚合缓存 (Aggregation Buffers)
    # numerator:   用于累加去噪后的像素值 (加权后)
    # denominator: 用于累加权重
    # 大小和 padding 后的图像一致
    numerator = np.zeros_like(img_pad, dtype=np.float32)
    denominator = np.zeros_like(img_pad, dtype=np.float32)

    # 4. 预计算 Kaiser 窗
    window = get_kaiser_window(p.patch_size)

    print(f"BM3D Stage 1 初始化完成:")
    print(f"  - 图像尺寸 (Padded): {img_pad.shape}")
    print(f"  - 块大小 (N1): {p.patch_size}")
    print(f"  - 搜索窗口 (Ns): {p.window_size}")
    print(f"  - 3D 阈值: {p.lambda_3d:.4f}")

    return img_pad, numerator, denominator, window, p, pad_w

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "data"
    noisy_path = DATA_DIR / "classic_photo_AWGN_sigma20_seed123456" / "lena_gray_sigma20_seed123456.png"
    original_path = DATA_DIR / "classic_photo" / "lena_gray.png"
    # 2. 读取图片
    original_img = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
    noisy_img = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
    if noisy_img is None:
        raise FileNotFoundError(f"找不到图片：{noisy_path}")

    # 3. 关键修正：保持 0-255 范围 (不要除以 255.0)
    # BM3D 的参数 (sigma=20, tau=2500) 都是针对 0-255 设计的
    noisy_img_float = noisy_img.astype(np.float32)

    # 4. 运行 BM3D Stage 1
    # 传入 sigma=20 (对应 0-255 范围的标准差)
    print(f"开始处理图片: {noisy_img_float.shape}, Sigma=20")
    sigma = 20
    patch_size = 8
    stride = patch_size // 2
    window_size = 39
    top_k = 16
    denoised_img = run_bm3d_stage1(noisy_img_float, sigma, patch_size, stride, window_size, top_k)

    # 5. 显示或保存结果 (可选)
    # 如果需要显示，记得裁剪回 0-255 并转为 uint8
    denoised_uint8 = np.clip(denoised_img, 0, 255).astype(np.uint8)
    cv2.imwrite("denoised_lena_stage1.png", denoised_uint8)
    print("处理完成，已保存为 denoised_lena_stage1.png")
    # 6. 计算指标 (Noise vs Original)
    print("-" * 30)
    noise_metrics = getMetrics.calculate_metrics(original_img.astype(np.uint8), noisy_img.astype(np.uint8))
    print(f"【带噪图】 PSNR: {noise_metrics['PSNR']:.2f} | SSIM: {noise_metrics['SSIM']:.4f}")

    # 7. 计算指标 (Denoised vs Original)
    denoised_metrics = getMetrics.calculate_metrics(original_img.astype(np.uint8), denoised_uint8)
    print(f"【降噪后】 PSNR: {denoised_metrics['PSNR']:.2f} | SSIM: {denoised_metrics['SSIM']:.4f}")
    print("-" * 30)

    lpips, _ = AI_Metrics.compare_advanced_metrics(str(original_path), str("denoised_lena_stage1.png"))
    print(f"{lpips:.4f} ")