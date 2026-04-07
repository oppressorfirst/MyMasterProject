"""
CRVD 数据集批量处理 - 官方 BM3D (VST 广义 Anscombe 变换策略)
  - 弃用粗糙的全局均值 Global Sigma 近似
  - 采用前向 VST -> 纯高斯域 BM3D (sigma=1.0) -> 后向代数逆 VST 的严谨流程
  - 输入/输出：RAW TIFF 图像 (包含归一化与反归一化)
"""

import os
import glob
import re
import csv
import time

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as skpsnr
from skimage.metrics import structural_similarity as skssim
from tqdm import tqdm

import bm3d

# ==========================================
# 1. 硬件/相机参数 (CRVD)
# ==========================================
BLACK = 240.0
WHITE = 4095.0
RANGE = WHITE - BLACK  # 3855.0

# ==========================================
# 2. VST 广义 Anscombe 变换 (修正代数逆公式)
# ==========================================
def forward_gat(z, a, sigma):
    return 2.0 * np.sqrt(np.maximum(z / a + 3.0 / 8.0 + (sigma ** 2) / (a ** 2), 0))

def inverse_gat(D, a, sigma):
    return a * ((D / 2.0) ** 2 - 1.0 / 8.0 - (sigma ** 2) / (a ** 2))

# ==========================================
# 3. Bayer 打包与解包
# ==========================================
def pack_bayer(raw_img: np.ndarray) -> np.ndarray:
    H, W = raw_img.shape
    packed = np.zeros((H // 2, W // 2, 4), dtype=raw_img.dtype)
    packed[:, :, 0] = raw_img[0::2, 0::2]
    packed[:, :, 1] = raw_img[0::2, 1::2]
    packed[:, :, 2] = raw_img[1::2, 0::2]
    packed[:, :, 3] = raw_img[1::2, 1::2]
    return packed

def unpack_bayer(packed: np.ndarray) -> np.ndarray:
    H2, W2, C = packed.shape
    raw_img = np.zeros((H2 * 2, W2 * 2), dtype=packed.dtype)
    raw_img[0::2, 0::2] = packed[:, :, 0]
    raw_img[0::2, 1::2] = packed[:, :, 1]
    raw_img[1::2, 0::2] = packed[:, :, 2]
    raw_img[1::2, 1::2] = packed[:, :, 3]
    return raw_img

# ==========================================
# 4. 噪声参数获取 (CRVD)
# ==========================================
def load_noise_params(csv_path: str) -> dict:
    """
    读取 CRVD 的噪声参数，并将 a 和 b 转换到 [0, 1] 归一化域。
    返回: {(scene, iso, frame): (beta1, beta2)}  -> (a_norm, b_norm)
    """
    params = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['scene'], int(row['iso']), int(row['frame']))
            if key not in params:
                a_raw = float(row['a'])
                b_raw = max(float(row['b']), 0.0)
                beta1 = a_raw / RANGE
                beta2 = b_raw / (RANGE ** 2)
                params[key] = (beta1, beta2)
    return params

# ==========================================
# 5. 单图处理逻辑 (核心修改区)
# ==========================================
def process_one_image(args: tuple) -> dict:
    noisy_path, clean_path, iso, scene, frame_id, noisy_idx, out_dir, beta1, beta2 = args

    # 1. 读图与归一化
    noisy_raw = cv2.imread(noisy_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    clean_raw = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    noisy_norm = np.clip((noisy_raw - BLACK) / RANGE, 0.0, 1.0)
    clean_norm = np.clip((clean_raw - BLACK) / RANGE, 0.0, 1.0)

    psnr_in = float(skpsnr(clean_norm, noisy_norm, data_range=1.0))

    # 2. Bayer 打包
    noisy_packed = pack_bayer(noisy_norm)
    denoised_packed = np.zeros_like(noisy_packed, dtype=np.float32)

    t0 = time.perf_counter()

    # 3. 逐通道 VST -> 官方 BM3D -> 逆 VST
    for ch in range(4):
        noisy_ch = noisy_packed[:, :, ch]
        
        # --- 核心修改：使用 VST 替代 Global Sigma ---
        # 3.1 前向 VST 变换
        noisy_vst = forward_gat(noisy_ch, beta1, beta2)

        # 3.2 官方 BM3D 处理 (VST域内噪声标准差已被稳定为近乎 1.0)
        sigma_vst = 1.0
        denoised_vst = bm3d.bm3d(noisy_vst, sigma_psd=sigma_vst, profile='np', 
                                 stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        
        # 3.3 后向逆 VST 变换
        denoised_ch = inverse_gat(denoised_vst, beta1, beta2)
        # ----------------------------------------
        
        denoised_packed[:, :, ch] = np.clip(denoised_ch, 0.0, 1.0)

    t_filter = time.perf_counter() - t0

    # 4. 解包与反归一化
    denoised_norm = unpack_bayer(denoised_packed)
    psnr_out = float(skpsnr(clean_norm, denoised_norm, data_range=1.0))
    ssim_out = float(skssim(clean_norm, denoised_norm, data_range=1.0))

    # 保存 TIFF (恢复到 16-bit)
    os.makedirs(out_dir, exist_ok=True)
    out_tiff = os.path.join(out_dir, f"frame{frame_id}_noisy{noisy_idx}_denoised.tiff")
    denoised_uint16 = np.clip(denoised_norm * RANGE + BLACK, 0.0, WHITE).astype(np.uint16)
    cv2.imwrite(out_tiff, denoised_uint16)

    return {
        'scene':            scene,
        'iso':              iso,
        'frame':            frame_id,
        'noisy_idx':        noisy_idx,
        'psnr_noisy_dB':    round(psnr_in,    4),
        'psnr_denoised_dB': round(psnr_out,   4),
        'ssim':             round(ssim_out,   4),
        'time_filter_s':    round(t_filter,   2),
        'beta1':            round(beta1,      8),
        'beta2':            round(beta2,     12),
    }

# ==========================================
# 6. 任务收集与主流程
# ==========================================
def collect_tasks(crvd_root, results_root, noise_params, iso_filter, scene_filter, noisy_filter):
    tasks = []
    for scene_dir in sorted(glob.glob(os.path.join(crvd_root, 'scene*'))):
        scene = os.path.basename(scene_dir)
        if scene_filter and scene != scene_filter:
            continue
            
        for iso_dir in sorted(glob.glob(os.path.join(scene_dir, 'ISO*'))):
            iso_name = os.path.basename(iso_dir)
            iso_val  = int(re.sub(r'\D', '', iso_name))
            if iso_filter and iso_val != iso_filter:
                continue
                
            out_dir = os.path.join(results_root, scene, iso_name)
            
            for clean_path in sorted(glob.glob(os.path.join(iso_dir, 'frame*_clean.tiff'))):
                m = re.search(r'frame(\d+)_clean\.tiff', os.path.basename(clean_path))
                if not m: continue
                fid = int(m.group(1))
                key = (scene, iso_val, fid)
                
                if key not in noise_params:
                    tqdm.write(f"[SKIP] 找不到噪声参数: {key}")
                    continue
                
                beta1, beta2 = noise_params[key]
                
                for np_path in sorted(glob.glob(os.path.join(iso_dir, f'frame{fid}_noisy*.tiff'))):
                    nm = re.search(r'frame\d+_noisy(\d+)\.tiff', os.path.basename(np_path))
                    nidx = int(nm.group(1)) if nm else -1
                    if noisy_filter is not None and nidx != noisy_filter:
                        continue
                        
                    tasks.append((
                        np_path, clean_path, iso_val, scene, fid, nidx,
                        out_dir, beta1, beta2
                    ))
    return tasks

def run(crvd_root, results_root, csv_out, params_csv, iso_filter, scene_filter, noisy_filter):
    print(f"加载噪声参数: {params_csv}")
    noise_params = load_noise_params(params_csv)
    tasks = collect_tasks(crvd_root, results_root, noise_params, iso_filter, scene_filter, noisy_filter)
    print(f"共收集到 {len(tasks)} 个图像任务。开始 VST+BM3D 处理...\n")

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    rows = []

    pbar = tqdm(tasks, desc='VST-BM3D-CRVD', dynamic_ncols=True)
    for task in pbar:
        try:
            row = process_one_image(task)
            rows.append(row)
            pbar.set_postfix({
                'Scene':    row['scene'],
                'ISO':      row['iso'],
                'f/n':      f"{row['frame']}/{row['noisy_idx']}",
                'PSNR↑':    f"{row['psnr_denoised_dB']:.2f} dB",
            })
        except Exception as e:
            tqdm.write(f"[ERR] {task[0]}: {e}")
    pbar.close()

    rows.sort(key=lambda r: (r['scene'], r['iso'], r['frame'], r['noisy_idx']))

    fieldnames = [
        'scene', 'iso', 'frame', 'noisy_idx',
        'psnr_noisy_dB', 'psnr_denoised_dB', 'ssim',
        'time_filter_s', 'beta1', 'beta2'
    ]
    with open(csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n✅ 结果已保存：{csv_out}  ({len(rows)} 行)")

    # 打印简要汇总统计
    from collections import defaultdict
    stats = defaultdict(list)
    for r in rows:
        stats[r['iso']].append((r['psnr_noisy_dB'], r['psnr_denoised_dB']))
    print(f"\n{'ISO':>8}  {'PSNR_noisy(dB)':>16}  {'PSNR_denoised(dB)':>18}  {'N':>5}")
    print('-' * 55)
    for iso in sorted(stats):
        arr = np.array(stats[iso])
        print(f"{iso:>8}  {np.mean(arr[:,0]):>16.3f}  "
              f"{np.mean(arr[:,1]):>18.3f}  {len(arr):>5}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Official BM3D with VST for CRVD")
    parser.add_argument('--root',   default='data/CRVD/noisy')
    parser.add_argument('--out',    default='out/results/CRVD/official_bm3d_vst')
    parser.add_argument('--csv',    default='out/results/CRVD/official_bm3d_vst/results.csv')
    parser.add_argument('--params', default='out/results/crvd_noise_params.csv')
    parser.add_argument('--iso', type=int, default=None)
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--noisy', type=int, default=None)
    args = parser.parse_args()

    run(args.root, args.out, args.csv, args.params, args.iso, args.scene, args.noisy)