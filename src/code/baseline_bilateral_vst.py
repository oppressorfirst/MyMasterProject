"""
CRVD 数据集 Bilateral Filter 降噪基线 - VST 版本
  - 噪声参数从 out/results/crvd_noise_params.csv 读取
  - 核心流程：前向 VST -> Bilateral Filter -> 后向逆 VST
  - 输出：denoised TIFF + PSNR CSV
      → out/results/CRVD/bilateral_vst/<scene>/<ISO>/
"""

from __future__ import annotations
import os
import glob
import re
import csv
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as skpsnr
from tqdm import tqdm

# ── 相机参数 ──────────────────────────────────────────────────────────────────
BLACK = 240.0
WHITE = 4095.0
RANGE = WHITE - BLACK   # 3855.0

# ── Bilateral 超参数 ──────────────────────────────────────────────────────────
# 在 VST 域内，噪声 sigma ≈ 1.0，sigmaColor 设为其倍数
BILATERAL_D          = 9     # 像素邻域直径
BILATERAL_SIGMA_COLOR = 2.0  # 颜色空间 sigma（VST 域 sigma≈1，取 2 倍留余量）
BILATERAL_SIGMA_SPACE = 9.0  # 空间坐标 sigma（与 D 相匹配）


# ── VST 相关函数 ───────────────────────────────────────────────────────────────

def forward_vst(x: np.ndarray, a: float, b: float) -> np.ndarray:
    c = 0.375 + b / (a ** 2)
    return 2.0 * np.sqrt(np.maximum(0.0, x / a + c))

def inverse_vst_algebraic(f: np.ndarray, a: float, b: float) -> np.ndarray:
    c = 0.375 + b / (a ** 2)
    return a * ((f ** 2) / 4.0 - c)


# ── 从 CSV 加载噪声参数 ────────────────────────────────────────────────────────

def load_noise_params(csv_path: str) -> dict[tuple[str, int, int], tuple[float, float]]:
    params: dict[tuple[str, int, int], tuple[float, float]] = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['scene'], int(row['iso']), int(row['frame']))
            if key not in params:
                a_raw = float(row['a'])
                b_raw = float(row['b'])
                params[key] = (a_raw / RANGE, b_raw / RANGE ** 2)
    return params


# ── 单图像处理 ────────────────────────────────────────────────────────────────

def process_one(args: tuple) -> dict:
    noisy_path, clean_path, iso, scene, frame_id, noisy_idx, out_dir, \
        a_norm, sigma_norm = args

    noisy_raw = cv2.imread(noisy_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    clean_raw = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    noisy = np.clip((noisy_raw - BLACK) / RANGE, 0.0, 1.0)
    clean = np.clip((clean_raw - BLACK) / RANGE, 0.0, 1.0)

    psnr_in = float(skpsnr(clean, noisy, data_range=1.0))

    # 1. 前向 VST：将 Poisson-Gaussian 噪声统一为 sigma≈1 的 AWGN
    noisy_vst = forward_vst(noisy, a_norm, sigma_norm)

    # 2. Bilateral Filter（VST 域，float32）
    t0 = time.perf_counter()
    denoised_vst = cv2.bilateralFilter(
        noisy_vst,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )
    t_denoise = time.perf_counter() - t0

    # 3. 后向逆 VST
    denoised = inverse_vst_algebraic(denoised_vst, a_norm, sigma_norm)
    denoised = np.clip(denoised, 0.0, 1.0)

    psnr_out = float(skpsnr(clean, denoised, data_range=1.0))

    denoised_raw = np.clip(denoised * RANGE + BLACK, 0.0, WHITE).astype(np.uint16)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"frame{frame_id}_noisy{noisy_idx}_denoised.tiff"),
                denoised_raw)

    return {
        'scene':            scene,
        'iso':              iso,
        'frame':            frame_id,
        'noisy_idx':        noisy_idx,
        'psnr_noisy_dB':    round(psnr_in,    4),
        'psnr_denoised_dB': round(psnr_out,   4),
        'time_s':           round(t_denoise,  4),
        'a_norm':           round(a_norm,      8),
        'sigma_norm':       round(sigma_norm, 12),
    }


# ── 任务收集 ───────────────────────────────────────────────────────────────────

def collect_tasks(
    crvd_root:    str,
    results_root: str,
    noise_params: dict[tuple[str, int, int], tuple[float, float]],
    iso_filter:   int | None = None,
    scene_filter: str | None = None,
    noisy_filter: int | None = None,
) -> list[tuple]:
    tasks: list[tuple] = []
    for scene_dir in sorted(glob.glob(os.path.join(crvd_root, 'scene*'))):
        scene = os.path.basename(scene_dir)
        if scene_filter is not None and scene != scene_filter:
            continue
        for iso_dir in sorted(glob.glob(os.path.join(scene_dir, 'ISO*'))):
            iso_name = os.path.basename(iso_dir)
            iso_val  = int(re.sub(r'\D', '', iso_name))
            if iso_filter is not None and iso_val != iso_filter:
                continue
            out_dir = os.path.join(results_root, scene, iso_name)
            for clean_path in sorted(glob.glob(
                    os.path.join(iso_dir, 'frame*_clean.tiff'))):
                m = re.search(r'frame(\d+)_clean\.tiff',
                              os.path.basename(clean_path))
                if not m:
                    continue
                fid = int(m.group(1))
                key = (scene, iso_val, fid)
                if key not in noise_params:
                    tqdm.write(f"[SKIP] 找不到噪声参数: {key}")
                    continue
                a_norm, sigma_norm = noise_params[key]
                for np_path in sorted(glob.glob(
                        os.path.join(iso_dir, f'frame{fid}_noisy*.tiff'))):
                    nm = re.search(r'frame\d+_noisy(\d+)\.tiff',
                                   os.path.basename(np_path))
                    nidx = int(nm.group(1)) if nm else -1
                    if noisy_filter is not None and nidx != noisy_filter:
                        continue
                    tasks.append((
                        np_path, clean_path, iso_val, scene, fid, nidx,
                        out_dir, a_norm, sigma_norm,
                    ))
    return tasks


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def run(
    crvd_root:    str = 'data/CRVD/noisy',
    results_root: str = 'out/results/CRVD/bilateral_vst',
    csv_out:      str = 'out/results/CRVD/bilateral_vst/results.csv',
    params_csv:   str = 'out/results/crvd_noise_params.csv',
    iso_filter:   int | None = None,
    scene_filter: str | None = None,
    noisy_filter: int | None = None,
    num_workers:  int = 8,
) -> None:

    print(f"加载噪声参数: {params_csv}")
    noise_params = load_noise_params(params_csv)
    print(f"  已读取 {len(noise_params)} 组 (scene, iso, frame) 参数")

    tasks = collect_tasks(crvd_root, results_root, noise_params,
                          iso_filter, scene_filter, noisy_filter)
    print(f"共 {len(tasks)} 张图像，跨图像使用 {num_workers} 核并行…\n")

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    rows: list[dict] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, t): t for t in tasks}
        pbar = tqdm(total=len(tasks), desc='Bilateral-VST', dynamic_ncols=True)
        for fut in futures:
            try:
                r = fut.result()
                rows.append(r)
                pbar.set_postfix({
                    'ISO':   r['iso'],
                    'f':     r['frame'],
                    'n':     r['noisy_idx'],
                    'PSNR↑': f"{r['psnr_denoised_dB']:.2f} dB",
                })
            except Exception as e:
                tqdm.write(f"[ERR] {e}")
            pbar.update(1)
        pbar.close()

    rows.sort(key=lambda r: (r['scene'], r['iso'], r['frame'], r['noisy_idx']))

    fieldnames = ['scene', 'iso', 'frame', 'noisy_idx',
                  'psnr_noisy_dB', 'psnr_denoised_dB', 'time_s',
                  'a_norm', 'sigma_norm']
    with open(csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n结果已保存：{csv_out}  ({len(rows)} 行)")

    stats: dict[int, list] = defaultdict(list)
    for r in rows:
        stats[r['iso']].append((r['psnr_noisy_dB'], r['psnr_denoised_dB']))

    print(f"\n{'ISO':>8}  {'PSNR_noisy(dB)':>16}  {'PSNR_denoised(dB)':>18}  {'N':>5}")
    print('-' * 55)
    for iso in sorted(stats):
        arr = np.array(stats[iso])
        print(f"{iso:>8}  {np.mean(arr[:, 0]):>16.3f}  "
              f"{np.mean(arr[:, 1]):>18.3f}  {len(arr):>5}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='CRVD Bilateral Filter 降噪基线（VST + 跨图像多进程）')
    parser.add_argument('--root',    default='data/CRVD/noisy')
    parser.add_argument('--out',     default='out/results/CRVD/bilateral_vst')
    parser.add_argument('--csv',     default='out/results/CRVD/bilateral_vst/results.csv')
    parser.add_argument('--params',  default='out/results/crvd_noise_params.csv')
    parser.add_argument('--iso',     type=int, default=None)
    parser.add_argument('--scene',   type=str, default=None)
    parser.add_argument('--noisy',   type=int, default=None)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    run(args.root, args.out, args.csv, args.params,
        args.iso, args.scene, args.noisy, args.workers)
