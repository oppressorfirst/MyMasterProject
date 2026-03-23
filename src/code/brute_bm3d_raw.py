import time
import pathlib
import h5py
import scipy.io as sio
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


# ==========================================
# 1. 广义 Anscombe 变换 (VST)
# ==========================================
def forward_gat(z, a, sigma):
    """将泊松-高斯混合噪声转化为近似标准差为 1 的高斯白噪声"""
    return 2.0 * np.sqrt(np.maximum(z / a + 3.0 / 8.0 + (sigma ** 2) / (a ** 2), 0))


def inverse_gat(D, a, sigma):
    """GAT 的渐近逆变换，映射回泊松-高斯域"""
    return a * ((D / 2.0) ** 2 - 1.0 / 8.0 - (sigma ** 2) / (a ** 2))


# ==========================================
# 2. 辅助函数：Bayer 打包与解包
# ==========================================
def pack_bayer(raw_img):
    H, W = raw_img.shape
    packed = np.zeros((H // 2, W // 2, 4), dtype=raw_img.dtype)
    packed[:, :, 0] = raw_img[0::2, 0::2]
    packed[:, :, 1] = raw_img[0::2, 1::2]
    packed[:, :, 2] = raw_img[1::2, 0::2]
    packed[:, :, 3] = raw_img[1::2, 1::2]
    return packed


def unpack_bayer(packed):
    H2, W2, C = packed.shape
    raw_img = np.zeros((H2 * 2, W2 * 2), dtype=packed.dtype)
    raw_img[0::2, 0::2] = packed[:, :, 0]
    raw_img[0::2, 1::2] = packed[:, :, 1]
    raw_img[1::2, 0::2] = packed[:, :, 2]
    raw_img[1::2, 1::2] = packed[:, :, 3]
    return raw_img


# ==========================================
# 3. 噪声参数获取 (SIDD 数据集)
# ==========================================
def get_noise_and_bayer_info(scene_id, noise_csv_path, bayer_csv_path):
    camera_id = scene_id.split('_')[2]
    df_bayer = pd.read_csv(bayer_csv_path)
    bayer_pattern = df_bayer[df_bayer['camera_id'] == camera_id]['bayer_pattern'].values[0].lower()

    color_map = {
        'rggb': ['r', 'g', 'g', 'b'],
        'bggr': ['b', 'g', 'g', 'r'],
        'grbg': ['g', 'r', 'b', 'g'],
        'gbrg': ['g', 'b', 'r', 'g']
    }
    channel_colors = color_map[bayer_pattern]

    df_noise = pd.read_csv(noise_csv_path)
    scene_row = df_noise[df_noise['scene_instance_id'] == scene_id].iloc[0]

    params = []
    for color in channel_colors:
        beta1 = scene_row[f'beta1_{color}']
        beta2 = scene_row[f'beta2_{color}']
        params.append((beta1, beta2))

    return params, bayer_pattern


# ==========================================
# 4. 核心算法：VST 域的 4 通道 BM3D 滤波
# ==========================================
def bm3d_1st_stage_vst_packed(img_vst_packed, all_results, patch_size):
    """
    在 VST 域执行的 4 通道 BM3D 滤波
    注意：因为传入的是 VST 图像，所以算法内部的 sigma 被硬编码为恒定的 1.0
    """
    H, W, C = img_vst_packed.shape
    numerator = np.zeros_like(img_vst_packed, dtype=np.float64)
    denominator = np.zeros_like(img_vst_packed, dtype=np.float64)

    # VST 域的最核心优势：全局唯一的 sigma = 1.0
    sigma_vst = 1.0
    lambda_3d = 2.7 * sigma_vst
    sigma_vst2 = sigma_vst ** 2

    print("Starting BM3D 1st Stage in VST domain (sigma=1.0) ...")
    for res in tqdm(all_results, desc="3D Filtering (VST)"):
        ref_y, ref_x = res["ref_pos"]
        matches = res["top_matches"]

        # 整理坐标与堆叠 4D 张量
        coords = [(ref_y, ref_x)] + [(y, x) for dist, y, x in matches]
        K_actual = len(coords)
        group_4d = np.zeros((K_actual, patch_size, patch_size, C), dtype=np.float64)
        for i, (y, x) in enumerate(coords):
            group_4d[i] = img_vst_packed[y:y + patch_size, x:x + patch_size, :]

        group_4d_denoised = np.zeros_like(group_4d)
        total_weight = 0

        # 对 4 个通道独立执行 3D 变换
        for ch in range(C):
            group_3d = group_4d[:, :, :, ch]

            # 3D 变换
            group_3d_freq = dctn(group_3d, norm='ortho')

            # 硬阈值截断 (使用固定的 2.7 * 1.0)
            group_3d_freq[np.abs(group_3d_freq) < lambda_3d] = 0

            # 计算权重 (使用统一的 sigma_vst2 = 1.0)
            n_nonzero = np.sum(group_3d_freq != 0)
            weight = 1.0 / (n_nonzero * sigma_vst2) if n_nonzero > 0 else 1.0 / sigma_vst2
            total_weight += weight

            # 逆 3D 变换
            group_4d_denoised[:, :, :, ch] = idctn(group_3d_freq, norm='ortho')

        # 聚合权重：4 通道平均
        avg_weight = total_weight / C

        for i, (y, x) in enumerate(coords):
            numerator[y:y + patch_size, x:x + patch_size, :] += group_4d_denoised[i] * avg_weight
            denominator[y:y + patch_size, x:x + patch_size, :] += avg_weight

    mask = denominator > 0
    denoised_vst_packed = img_vst_packed.copy()
    denoised_vst_packed[mask] = numerator[mask] / denominator[mask]

    return denoised_vst_packed


if __name__ == "__main__":
    # =========================
    # Step 1 - 配置与读取真实数据
    # =========================
    dataset_path = "data/SIDD_small_RAW/"
    SCENE_NAME = "0065_003_GP_10000_08460_4400_N"
    scene_dir = pathlib.Path(dataset_path) / SCENE_NAME

    NOISE_CSV = pathlib.Path(dataset_path) / "noise_level_functions.csv"
    BAYER_CSV = pathlib.Path(dataset_path) / "bayer_patterns.csv"
    GT_MAT = scene_dir / "GT_RAW_010.mat"
    NOISY_MAT = scene_dir / "NOISY_RAW_010.mat"

    print(f"Loading real SIDD data: {SCENE_NAME} ...")
    with h5py.File(GT_MAT, 'r') as f:
        gt_raw = np.array(f['x']).T.astype(np.float32)
    with h5py.File(NOISY_MAT, 'r') as f:
        noisy_raw = np.array(f['x']).T.astype(np.float32)

    max_val = max(np.max(gt_raw), 1.0)
    if max_val > 10.0:
        gt_raw = gt_raw / max_val
        noisy_raw = noisy_raw / max_val

    # 获取真实参数并 Pack 成 4 通道
    channel_params, bayer_pattern = get_noise_and_bayer_info(SCENE_NAME, NOISE_CSV, BAYER_CSV)
    gt_packed = pack_bayer(gt_raw)
    noisy_packed = pack_bayer(noisy_raw)

    # =======================================================
    # Step 2 - 切片处理 (Crop) 以加速验证
    # =======================================================
    # H_p, W_p, C = noisy_packed.shape
    # crop_size = 256
    # start_y = H_p // 2 - crop_size // 2
    # start_x = W_p // 2 - crop_size // 2
    #
    # noisy_packed_crop = noisy_packed[start_y:start_y + crop_size, start_x:start_x + crop_size, :].copy()
    # gt_packed_crop = gt_packed[start_y:start_y + crop_size, start_x:start_x + crop_size, :].copy()
    # H_crop, W_crop, _ = noisy_packed_crop.shape
    # =======================================================
    # Step 2 - 全图处理 (取消 Crop)
    # =======================================================
    H_full, W_full, C = noisy_packed.shape

    # 直接使用完整的 Packed 图像
    noisy_packed_full = noisy_packed.copy()
    gt_packed_full = gt_packed.copy()

    # 算法参数保持不变
    patch_size = 8
    top_k = 16
    window_size = 39
    stride = patch_size // 2
    search_radius = window_size // 2

    # 后续代码中的 H_crop, W_crop 变量替换为全图的高和宽
    H_crop, W_crop = H_full, W_full

    # 注意：后续处理使用的变量名我也帮你保留了 noisy_packed_crop / gt_packed_crop 的命名，
    # 这样你就不用去修改下面 Step 3 到 Step 6 的变量名了。
    noisy_packed_crop = noisy_packed_full
    gt_packed_crop = gt_packed_full

    # =======================================================
    # Step 3 - 执行 Forward VST
    # =======================================================
    print("\n--- Applying Forward VST ---")
    vst_packed_crop = np.zeros_like(noisy_packed_crop)
    for ch in range(C):
        beta1, beta2 = channel_params[ch]
        sigma_est = np.sqrt(max(beta2, 0.0))
        # 逐通道进行 VST 变换
        vst_packed_crop[:, :, ch] = forward_gat(noisy_packed_crop[:, :, ch], a=beta1, sigma=sigma_est)

    # 算法参数
    patch_size = 8
    top_k = 16
    window_size = 39
    stride = patch_size // 2
    search_radius = window_size // 2

    # =========================
    # Step 4 - 在 VST 域执行暴力匹配
    # =========================
    # 注意：现在 Guide 图是在 VST 域上做的模糊
    guide_vst_packed = cv2.GaussianBlur(vst_packed_crop, (5, 5), 1.5)

    all_results = []
    ref_ys = list(range(0, H_crop - patch_size + 1, stride))
    ref_xs = list(range(0, W_crop - patch_size + 1, stride))
    total = len(ref_ys) * len(ref_xs)

    t_global0 = time.perf_counter()
    pbar = tqdm(total=total, desc="VST Domain Brute-force matching", dynamic_ncols=True)

    for ref_y in ref_ys:
        for ref_x in ref_xs:
            ref_patch = guide_vst_packed[ref_y:ref_y + patch_size, ref_x:ref_x + patch_size, :]
            matches = []

            y_min = max(0, ref_y - search_radius)
            y_max = min(H_crop - patch_size, ref_y + search_radius)
            x_min = max(0, ref_x - search_radius)
            x_max = min(W_crop - patch_size, ref_x + search_radius)

            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    if y == ref_y and x == ref_x:
                        continue
                    cand_patch = guide_vst_packed[y:y + patch_size, x:x + patch_size, :]

                    diff = ref_patch - cand_patch
                    dist = np.mean(diff * diff)
                    matches.append((dist, y, x))

            matches.sort(key=lambda t: t[0])
            top_matches = matches[:top_k]

            all_results.append({
                "ref_pos": (ref_y, ref_x),
                "top_matches": top_matches
            })
            pbar.update(1)

    pbar.close()
    print(f"Total VST RAW matching time: {time.perf_counter() - t_global0:.3f} s")

    # =========================
    # Step 5 - VST 域的 BM3D Filtering
    # =========================
    t_filter0 = time.perf_counter()
    # 传入 VST 变换后的图像
    denoised_vst_packed_crop = bm3d_1st_stage_vst_packed(vst_packed_crop, all_results, patch_size)
    print(f"Total 3D Filtering time: {time.perf_counter() - t_filter0:.3f} s")

    # =========================
    # Step 6 - 执行 Inverse VST 并恢复
    # =========================
    print("--- Applying Inverse VST ---")
    denoised_packed_crop = np.zeros_like(denoised_vst_packed_crop)
    for ch in range(C):
        beta1, beta2 = channel_params[ch]
        sigma_est = np.sqrt(max(beta2, 0.0))
        # 逐通道逆变换回泊松-高斯域
        denoised_packed_crop[:, :, ch] = inverse_gat(denoised_vst_packed_crop[:, :, ch], a=beta1, sigma=sigma_est)

    # 限制范围防溢出
    denoised_packed_crop = np.clip(denoised_packed_crop, 0.0, 1.0)

    # 解包与指标评估
    gt_raw_crop = unpack_bayer(gt_packed_crop)
    denoised_raw_crop = unpack_bayer(denoised_packed_crop)

    current_psnr = psnr(gt_raw_crop, denoised_raw_crop, data_range=1.0)
    current_ssim = ssim(gt_raw_crop, denoised_raw_crop, data_range=1.0)
    print(f"\nFinal VST CROP RAW PSNR: {current_psnr:.2f} dB | SSIM: {current_ssim:.4f}\n")

    # 可视化评估
    noisy_raw_crop = unpack_bayer(noisy_packed_crop)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Clean RAW (Crop, Top-Left Pixel)")
    plt.imshow(gt_raw_crop[0::2, 0::2], cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Noisy RAW (Crop)")
    plt.imshow(noisy_raw_crop[0::2, 0::2], cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Denoised RAW (VST + BM3D)")
    plt.imshow(denoised_raw_crop[0::2, 0::2], cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.show()