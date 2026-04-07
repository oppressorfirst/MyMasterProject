"""
CRVD 数据集批量 BM3D 降噪（泊松-高斯自适应）- 单图内层 40 核并行版
  - 噪声参数从 out/results/crvd_noise_params.csv 读取（per-scene/iso/frame）
  - 外层循环逐图顺序处理，内层 ProcessPoolExecutor(40 核) 并行执行块匹配
  - 向量化块匹配结合 Chunking 分发，降低 IPC 序列化开销
  - 输出：denoised TIFF + PSNR CSV
      → out/results/CRVD/my_brute_bm3d_8c/<scene>/<ISO>/
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

# ── 相机参数（与 crvd_raw2png.py 一致）──────────────────────────────────────
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


# ── 从 CSV 加载噪声参数 ────────────────────────────────────────────────────────

def load_noise_params(
    csv_path: str,
) -> dict[tuple[str, int, int], tuple[float, float]]:
    """
    读取 crvd_noise_params.csv，返回
        (scene, iso, frame) -> (a_norm, sigma_norm)

    RAW 域模型:   Var_raw = a_raw*(x_raw - BLACK) + b_raw
    归一化转换:    a_norm    = a_raw / RANGE
                  sigma_norm = b_raw / RANGE²   （高斯方差分量）

    同一 (scene, iso, frame) 对应多条 noisy_idx 行，但 a/b 值相同，
    只取第一次出现的值。
    """
    params: dict[tuple[str, int, int], tuple[float, float]] = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['scene'], int(row['iso']), int(row['frame']))
            if key not in params:
                a_raw = float(row['a'])
                b_raw = float(row['b'])
                params[key] = (a_raw / RANGE, b_raw / RANGE ** 2)
    return params


# ── 独立的工作进程函数（必须定义在顶层以支持 Pickle 序列化）─────────────────────

def _match_chunk(
    coords_chunk: list[tuple[int, int]],
    guide: np.ndarray,
    rad: int,
    patch_size: int,
    top_k: int,
) -> list[dict]:
    """
    工作进程函数：处理分配到的一批参考块坐标，执行向量化滑动窗口匹配。
    """
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

        # 排除自身
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
    """
    将参考块坐标切分为 num_workers 个 chunk，分发给进程池并行执行块匹配。
    """
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


# ── BM3D 自适应硬阈值协同滤波 ─────────────────────────────────────────────────

def bm3d_poisson_gaussian(img: np.ndarray, all_results: list[dict],
                          a: float, sigma_norm: float) -> np.ndarray:
    """
    泊松-高斯自适应 BM3D 第一阶段（硬阈值）。
        Var_local = a * mean_local + sigma_norm
        lambda    = 2.7 * sqrt(Var_local)
    """
    H, W = img.shape
    num = np.zeros((H, W), dtype=np.float64)
    den = np.zeros((H, W), dtype=np.float64)

    for res in all_results:
        ry, rx = res['ref_pos']
        coords = [(ry, rx)] + [(y, x) for _, y, x in res['top_matches']]

        group = np.stack(
            [img[y:y + PATCH, x:x + PATCH] for y, x in coords], axis=0
        ).astype(np.float64)

        local_mean = max(float(group[0].mean()), 0.0)
        local_var  = a * local_mean + sigma_norm
        threshold  = 2.7 * np.sqrt(max(local_var, 1e-12))

        freq = dctn(group, norm='ortho')
        freq[np.abs(freq) < threshold] = 0.0
        n_nz = int(np.count_nonzero(freq))
        w    = 1.0 / (n_nz * local_var) if n_nz > 0 else 1.0 / max(local_var, 1e-12)

        denoised = idctn(freq, norm='ortho')
        for i, (y, x) in enumerate(coords):
            num[y:y + PATCH, x:x + PATCH] += denoised[i] * w
            den[y:y + PATCH, x:x + PATCH] += w

    out = img.copy()
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return np.clip(out, 0.0, 1.0)


# ── 单图像处理流程 ────────────────────────────────────────────────────────────

def process_one(args: tuple) -> dict:
    """
    读取单张图 -> 块匹配 (40核并行) -> BM3D滤波 -> 保存 TIFF -> 返回指标。
    args: (noisy_path, clean_path, iso, scene, frame_id, noisy_idx, out_dir,
           a_norm, sigma_norm)
    """
    noisy_path, clean_path, iso, scene, frame_id, noisy_idx, out_dir, \
        a_norm, sigma_norm = args

    noisy_raw = cv2.imread(noisy_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    clean_raw = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    noisy = np.clip((noisy_raw - BLACK) / RANGE, 0.0, 1.0)
    clean = np.clip((clean_raw - BLACK) / RANGE, 0.0, 1.0)

    psnr_in = float(skpsnr(clean, noisy, data_range=1.0))

    guide = cv2.GaussianBlur(noisy, (5, 5), 1.5)

    t0 = time.perf_counter()
    matches = vectorized_match_parallel(guide, num_workers=NUM_WORKERS)
    t_match = time.perf_counter() - t0

    t1 = time.perf_counter()
    denoised = bm3d_poisson_gaussian(noisy, matches, a_norm, sigma_norm)
    t_filter = time.perf_counter() - t1

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
) -> list[tuple]:
    """
    遍历 CRVD 目录，为每张 noisy TIFF 构建一个任务 tuple，
    其中包含从 noise_params 中查到的 (a_norm, sigma_norm)。
    """
    tasks: list[tuple] = []
    for scene_dir in sorted(glob.glob(os.path.join(crvd_root, 'scene*'))):
        scene = os.path.basename(scene_dir)
        for iso_dir in sorted(glob.glob(os.path.join(scene_dir, 'ISO*'))):
            iso_name = os.path.basename(iso_dir)
            iso_val  = int(re.sub(r'\D', '', iso_name))
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
                    tasks.append((
                        np_path, clean_path, iso_val, scene, fid, nidx,
                        out_dir, a_norm, sigma_norm,
                    ))
    return tasks


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def run(
    crvd_root:    str = 'data/CRVD/noisy',
    results_root: str = 'out/results/CRVD/my_brute_bm3d_no_VST',
    csv_out:      str = 'out/results/CRVD/my_brute_bm3d_no_VST/results.csv',
    params_csv:   str = 'out/results/crvd_noise_params.csv',
) -> None:

    print(f"加载噪声参数: {params_csv}")
    noise_params = load_noise_params(params_csv)
    print(f"  已读取 {len(noise_params)} 组 (scene, iso, frame) 参数")

    tasks = collect_tasks(crvd_root, results_root, noise_params)
    print(f"共 {len(tasks)} 张图像，单图内层使用 {NUM_WORKERS} 核并行块匹配…\n")

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    rows: list[dict] = []

    pbar = tqdm(tasks, desc='BM3D-CRVD', dynamic_ncols=True)
    for task in pbar:
        try:
            row = process_one(task)
            rows.append(row)
            pbar.set_postfix({
                'ISO':      row['iso'],
                'f':        row['frame'],
                'n':        row['noisy_idx'],
                'PSNR↑':    f"{row['psnr_denoised_dB']:.2f} dB",
                'match(s)': row['time_match_s'],
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
        description='CRVD BM3D 批量降噪（泊松-高斯自适应，40核单图并行）')
    parser.add_argument('--root',   default='data/CRVD/noisy',
                        help='CRVD noisy 根目录')
    parser.add_argument('--out',    default='out/results/CRVD/my_brute_bm3d_8c',
                        help='输出根目录')
    parser.add_argument('--csv',    default='out/results/CRVD/my_brute_bm3d_8c/results.csv',
                        help='PSNR 结果 CSV 路径')
    parser.add_argument('--params', default='out/results/crvd_noise_params.csv',
                        help='噪声参数 CSV（由 crvd_noise_estimate.py 生成）')
    args = parser.parse_args()

    run(args.root, args.out, args.csv, args.params)
