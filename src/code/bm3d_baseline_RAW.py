import pathlib
import h5py
import numpy as np
import scipy.io as sio
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 引入官方 bm3d 库
import bm3d


# ==========================================
# 1. Bayer 打包与解包
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
# 2. 噪声参数获取
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
        a_est = beta1
        sigma_est = np.sqrt(max(beta2, 0.0))
        params.append((a_est, sigma_est))

    return params, bayer_pattern


# ==========================================
# 3. 主流程 (策略1：无 VST，使用全局标量 Sigma)
# ==========================================
def process_baseline_global_sigma(gt_path, noisy_path, scene_id, noise_csv, bayer_csv, out_path):
    print(f"Loading data for scene: {scene_id} ...")

    # 1. 读取 MAT 文件
    with h5py.File(gt_path, 'r') as f:
        gt_raw = np.array(f['x']).T.astype(np.float32)
    with h5py.File(noisy_path, 'r') as f:
        noisy_raw = np.array(f['x']).T.astype(np.float32)

    # 归一化判断
    max_val = max(np.max(gt_raw), 1.0)
    if max_val > 10.0:
        gt_raw = gt_raw / max_val
        noisy_raw = noisy_raw / max_val

    # 2. 获取噪声参数
    channel_params, bayer_pattern = get_noise_and_bayer_info(scene_id, noise_csv, bayer_csv)
    print(f"Bayer Pattern: {bayer_pattern.upper()}")

    # 3. 打包为 4 通道
    gt_packed = pack_bayer(gt_raw)
    noisy_packed = pack_bayer(noisy_raw)
    denoised_packed = np.zeros_like(noisy_packed)

    # 4. 逐通道独立处理 (剥离 VST)
    for ch in range(4):
        a_est, sigma_est = channel_params[ch]
        beta1 = a_est
        beta2 = sigma_est ** 2

        clean_ch = gt_packed[:, :, ch]
        noisy_ch = noisy_packed[:, :, ch]

        # === 核心修改：基于当前通道的全局均值计算标量 Sigma ===
        # 计算整张图（通道）的亮度均值
        mean_val = np.mean(noisy_ch)

        # 套用物理公式: Var = beta1 * I_mean + beta2
        global_variance = max(beta1 * mean_val + beta2, 1e-10)
        global_sigma = np.sqrt(global_variance)

        print(f"\n--- Running Official BM3D (Global Sigma) for Channel {ch + 1}/4 ---")
        print(f"    beta1={beta1:.6e}, beta2={beta2:.6e}")
        print(f"    Channel Mean={mean_val:.4f} -> Global Sigma={global_sigma:.6f}")

        # 运行官方 BM3D，传入计算好的单一标量 global_sigma
        denoised_ch = bm3d.bm3d(noisy_ch, sigma_psd=global_sigma, profile='np',
                                stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

        # 限制范围
        denoised_packed[:, :, ch] = np.clip(denoised_ch, 0.0, 1.0)

    # 5. 解包还原为完整 RAW 图
    denoised_raw = unpack_bayer(denoised_packed)

    # 6. 计算指标
    psnr_val = psnr(gt_raw, denoised_raw, data_range=1.0)
    ssim_val = ssim(gt_raw, denoised_raw, data_range=1.0)

    print("\n================ OFFICIAL BM3D (GLOBAL SIGMA) RESULTS ================")
    print(f"PSNR: {psnr_val:.4f} dB")
    print(f"SSIM: {ssim_val:.4f}")

    # 7. 导出到 MAT 文件
    sio.savemat(out_path, {'x': denoised_raw})
    print(f"Baseline Denoised RAW (Global Sigma) saved to {out_path}")

    return denoised_raw, psnr_val, ssim_val


if __name__ == "__main__":
    dataset_path = "data/SIDD_small_RAW/"
    SCENE_NAME = "0065_003_GP_10000_08460_4400_N"

    scene_dir = pathlib.Path(dataset_path) / SCENE_NAME

    NOISE_CSV = pathlib.Path(dataset_path) / "noise_level_functions.csv"
    BAYER_CSV = pathlib.Path(dataset_path) / "bayer_patterns.csv"

    GT_MAT = scene_dir / "GT_RAW_010.mat"
    NOISY_MAT = scene_dir / "NOISY_RAW_010.mat"

    OUT_MAT = scene_dir / "DENOISED_RAW_OFFICIAL_BM3D_GLOBAL_SIGMA.mat"

    process_baseline_global_sigma(GT_MAT, NOISY_MAT, SCENE_NAME, NOISE_CSV, BAYER_CSV, OUT_MAT)