"""
CRVD 数据集批量 BM3D 降噪（包含 VST 变换版本）- 单图内层 40 核并行版
  - 噪声参数从 out/results/crvd_noise_params.csv 读取
  - 核心流程：前向 VST -> 纯高斯域 BM3D (固定阈值) -> 后向逆 VST
  - 输出：denoised TIFF + PSNR CSV
      → out/results/CRVD/my_brute_bm3d_8c_vst/<scene>/<ISO>/
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
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio as skpsnr
from tqdm import tqdm

# ── 相机参数 ──────────────────────────────────────────────────────────────────
BLACK = 240.0
WHITE = 4095.0
RANGE = WHITE - BLACK   # 3855.0

# ── 并行核数 ──────────────────────────────────────────────────────────────────
NUM_WORKERS = 40

# ── BM3D 超参数 ────────────────────────────────────────────────────────────────
PATCH  = 8
TOP_K  = 8
WIN    = 39
STRIDE = PATCH // 2   # 4


# ── VST (广义 Anscombe 变换) 相关函数 ─────────────────────────────────────────

def forward_vst(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    前向广义 Anscombe 变换 (Generalized Anscombe Transform)
    将 Poisson-Gaussian 噪声 (Var = a*x + b) 转换为标准差近似为 1 的高斯白噪声。
    公式: f(x) = 2 * sqrt( max(0, x/a + 3/8 + b/a^2) )
    """
    c = 0.375 + b / (a ** 2)
    val = np.maximum(0.0, x / a + c)
    return 2.0 * np.sqrt(val)

def inverse_vst_algebraic(f: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    代数逆变换 (Algebraic Inverse)
    公式: I(f) = a * (f^2 / 4 - 3/8 - b/a^2)
    说明: 此版本为代数闭合解，不涉及复杂的除法和浮点次幂，非常适合 ASIC 的 RTL 实现。
    """
    c = 0.375 + b / (a ** 2)
    return a * ( (f ** 2) / 4.0 - c )


# ── 从 CSV 加载噪声参数 ────────────────────────────────────────────────────────

def load_noise_params(
    csv_path: str,
) -> dict[tuple[str, int, int], tuple[float, float]]:
    params: dict[tuple[str, int, int], tuple[float, float]] = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['scene'], int(row['iso']), int(row['frame']))
            if key not in params:
                a_raw = float(row['a'])
                b_raw = float(row['b'])
                params[key] = (a_raw / RANGE, b_raw / RANGE ** 2)
    return params


# ── 独立的工作进程函数 ─────────────────────────────────────────────────────────

def _match_chunk(
    coords_chunk: list[tuple[int, int]],
    guide: np.ndarray,
    rad: int,
    patch_size: int,
    top_k: int,
) -> list[dict]:
    H, W = guide.shape
    chunk_results: list[dict] = []

    for (ry, rx) in coords_chunk:
        ref = guide[ry:ry + patch_size, rx:rx + patch_size]

        y0 = max(0, ry - rad);   y1 = min(H - patch_size, ry + rad)
        x0 = max(0, rx - rad);   x1 = min(W - patch_size, rx + rad)

        region  = guide[y0:y1 + patch_size, x0:x1 + patch_size]
        patches = sliding_window_view(region, (patch_size, patch_size))
        ny, nx  = patches.shape[:2]

        dists = np.mean((patches - ref) ** 2, axis=(-2, -1))

        sy = ry - y0;  sx = rx - x0
        if 0 <= sy < ny and 0 <= sx < nx:
            dists[sy, sx] = np.inf

        YY, XX = np.meshgrid(
            np.arange(y0, y0 + ny),
            np.arange(x0, x0 + nx),
            indexing='ij',
        )
        flat_d = dists.ravel()
        flat_y = YY.ravel()
        flat_x = XX.ravel()

        valid  = np.isfinite(flat_d)
        flat_d = flat_d[valid]
        flat_y = flat_y[valid]
        flat_x = flat_x[valid]

        k = min(top_k, len(flat_d))
        if k == 0:
            chunk_results.append({'ref_pos': (ry, rx), 'top_matches': []})
            continue

        idx = np.argpartition(flat_d, k - 1)[:k]
        idx = idx[np.argsort(flat_d[idx])]
        top = [(float(flat_d[i]), int(flat_y[i]), int(flat_x[i])) for i in idx]
        chunk_results.append({'ref_pos': (ry, rx), 'top_matches': top})

    return chunk_results


# ── 向量化多进程块匹配 ────────────────────────────────────────────────────────

def vectorized_match_parallel(guide: np.ndarray,
                              num_workers: int = NUM_WORKERS) -> list[dict]:
    H, W = guide.shape
    rad = WIN // 2

    coords = [
        (ry, rx)
        for ry in range(0, H - PATCH + 1, STRIDE)
        for rx in range(0, W - PATCH + 1, STRIDE)
    ]

    chunk_size = max(1, len(coords) // num_workers)
    chunks = [coords[i:i + chunk_size] for i in range(0, len(coords), chunk_size)]

    all_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_match_chunk, chunk, guide, rad, PATCH, TOP_K)
            for chunk in chunks
        ]
        for fut in futures:
            all_results.extend(fut.result())

    return all_results


# ── BM3D 纯高斯白噪声协同滤波 (AWGN 模式) ─────────────────────────────────────

def bm3d_awgn_filtering(img_vst: np.ndarray, all_results: list[dict]) -> np.ndarray:
    """
    因为 VST 变换已经将噪声统一为 sigma=1.0 的白噪声，
    此处的滤波不再需要计算局部均值和方差，直接使用固定阈值。
    """
    H, W = img_vst.shape
    num = np.zeros((H, W), dtype=np.float64)
    den = np.zeros((H, W), dtype=np.float64)

    # 经过 VST 后，全局方差固定为 1.0
    sigma_vst = 1.0
    threshold = 2.7 * sigma_vst

    for res in all_results:
        ry, rx = res['ref_pos']
        coords = [(ry, rx)] + [(y, x) for _, y, x in res['top_matches']]

        group = np.stack(
            [img_vst[y:y + PATCH, x:x + PATCH] for y, x in coords], axis=0
        ).astype(np.float64)

        freq = dctn(group, norm='ortho')
        
        # 固定的全局硬阈值截断
        freq[np.abs(freq) < threshold] = 0.0
        n_nz = int(np.count_nonzero(freq))
        
        # 权重计算仅与非零系数数量有关，不依赖局部亮度
        w = 1.0 / (n_nz * (sigma_vst ** 2)) if n_nz > 0 else 1.0 / (sigma_vst ** 2)

        denoised = idctn(freq, norm='ortho')
        for i, (y, x) in enumerate(coords):
            num[y:y + PATCH, x:x + PATCH] += denoised[i] * w
            den[y:y + PATCH, x:x + PATCH] += w

    out = img_vst.copy()
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


# ── 单图像处理流程 ────────────────────────────────────────────────────────────

def process_one(args: tuple) -> dict:
    noisy_path, clean_path, iso, scene, frame_id, noisy_idx, out_dir, \
        a_norm, sigma_norm = args

    noisy_raw = cv2.imread(noisy_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    clean_raw = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    noisy = np.clip((noisy_raw - BLACK) / RANGE, 0.0, 1.0)
    clean = np.clip((clean_raw - BLACK) / RANGE, 0.0, 1.0)

    psnr_in = float(skpsnr(clean, noisy, data_range=1.0))

    # ==========================
    # 1. 前向 VST 变换
    # ==========================
    noisy_vst = forward_vst(noisy, a_norm, sigma_norm)

    # 引导图也在 VST 域生成
    guide_vst = cv2.GaussianBlur(noisy_vst, (5, 5), 1.5)

    # ==========================
    # 2. VST 域内的块匹配
    # ==========================
    t0 = time.perf_counter()
    matches = vectorized_match_parallel(guide_vst, num_workers=NUM_WORKERS)
    t_match = time.perf_counter() - t0

    # ==========================
    # 3. VST 域内的 BM3D 滤波 (固定阈值)
    # ==========================
    t1 = time.perf_counter()
    denoised_vst = bm3d_awgn_filtering(noisy_vst, matches)
    t_filter = time.perf_counter() - t1

    # ==========================
    # 4. 后向逆 VST 变换
    # ==========================
    denoised = inverse_vst_algebraic(denoised_vst, a_norm, sigma_norm)
    denoised = np.clip(denoised, 0.0, 1.0)

    psnr_out = float(skpsnr(clean, denoised, data_range=1.0))

    denoised_raw = np.clip(denoised * RANGE + BLACK, 0.0, WHITE).astype(np.uint16)
    os.makedirs(out_dir, exist_ok=True)
    out_tiff = os.path.join(out_dir, f"frame{frame_id}_noisy{noisy_idx}_denoised.tiff")
    cv2.imwrite(out_tiff, denoised_raw)

    return {
        'scene':            scene,
        'iso':              iso,
        'frame':            frame_id,
        'noisy_idx':        noisy_idx,
        'psnr_noisy_dB':    round(psnr_in,    4),
        'psnr_denoised_dB': round(psnr_out,   4),
        'time_match_s':     round(t_match,    1),
        'time_filter_s':    round(t_filter,   1),
        'a_norm':           round(a_norm,      8),
        'sigma_norm':       round(sigma_norm, 12),
    }


# ── 任务收集 ───────────────────────────────────────────────────────────────────

def collect_tasks(
    crvd_root: str,
    results_root: str,
    noise_params: dict[tuple[str, int, int], tuple[float, float]],
    iso_filter: int | None = None,
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
            out_dir  = os.path.join(results_root, scene, iso_name)
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
    results_root: str = 'out/results/CRVD/my_brute_bm3d_8c_vst',
    csv_out:      str = 'out/results/CRVD/my_brute_bm3d_8c_vst/results.csv',
    params_csv:   str = 'out/results/crvd_noise_params.csv',
    iso_filter:   int | None = None,
    scene_filter: str | None = None,
    noisy_filter: int | None = None,
) -> None:

    print(f"加载噪声参数: {params_csv}")
    noise_params = load_noise_params(params_csv)
    print(f"  已读取 {len(noise_params)} 组 (scene, iso, frame) 参数")

    tasks = collect_tasks(crvd_root, results_root, noise_params, iso_filter, scene_filter, noisy_filter)
    print(f"共 {len(tasks)} 张图像，单图内层使用 {NUM_WORKERS} 核并行块匹配…\n")

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    rows: list[dict] = []

    pbar = tqdm(tasks, desc='BM3D-CRVD-VST', dynamic_ncols=True)
    for task in pbar:
        try:
            row = process_one(task)
            rows.append(row)
            pbar.set_postfix({
                'ISO':      row['iso'],
                'f':        row['frame'],
                'n':        row['noisy_idx'],
                'PSNR↑':    f"{row['psnr_denoised_dB']:.2f} dB",
            })
        except Exception as e:
            tqdm.write(f"[ERR] {task[0]}: {e}")
    pbar.close()

    rows.sort(key=lambda r: (r['scene'], r['iso'], r['frame'], r['noisy_idx']))

    fieldnames = [
        'scene', 'iso', 'frame', 'noisy_idx',
        'psnr_noisy_dB', 'psnr_denoised_dB',
        'time_match_s', 'time_filter_s',
        'a_norm', 'sigma_norm',
    ]
    with open(csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n结果已保存：{csv_out}  ({len(rows)} 行)")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
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
        description='CRVD BM3D 批量降噪（VST 泊松高斯自适应，40核并行）')
    parser.add_argument('--root',   default='data/CRVD/noisy',
                        help='CRVD noisy 根目录')
    parser.add_argument('--out',    default='out/results/CRVD/my_brute_bm3d_8c_vst',
                        help='输出根目录')
    parser.add_argument('--csv',    default='out/results/CRVD/my_brute_bm3d_8c_vst/results.csv',
                        help='PSNR 结果 CSV 路径')
    parser.add_argument('--params', default='out/results/crvd_noise_params.csv',
                        help='噪声参数 CSV（由 crvd_noise_estimate.py 生成）')
    parser.add_argument('--iso', type=int, default=None,
                        help='只处理指定 ISO（如 25600），默认处理全部')
    parser.add_argument('--scene', type=str, default=None,
                        help='只处理指定 scene（如 scene1），默认处理全部')
    parser.add_argument('--noisy', type=int, default=None,
                        help='只处理指定 noisy 版本（如 0），默认处理全部')
    args = parser.parse_args()

    run(args.root, args.out, args.csv, args.params, args.iso, args.scene, args.noisy)