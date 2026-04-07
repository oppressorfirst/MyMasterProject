"""
CRVD 数据集批量 V-BM3D 降噪（包含 VST 变换版本）- 序列级 40 核并行版
  - 扩展：引入时间维度，跨帧搜索相似块 (Spatio-temporal Block Matching)
  - 核心流程：前向 VST (逐帧) -> 跨帧块匹配 -> 3D 纯高斯域协同滤波 -> 后向逆 VST (逐帧)
  - 内存说明：此版本为暴力验证版，会将整个时域窗口帧读入内存进行全局聚合。
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
RANGE = WHITE - BLACK

# ── 并行与超参数 ──────────────────────────────────────────────────────────────
NUM_WORKERS = 40

# V-BM3D 独有：时间窗口搜索半径。1 表示搜索前后各1帧 (即 t-1, t, t+1)
TEMP_RAD = 1  

PATCH  = 8
TOP_K  = 8     # 注意：此时 K 个相似块是从多个帧中共同挑出的最优解
WIN    = 39
STRIDE = PATCH // 2 

# ── VST (广义 Anscombe 变换) ──────────────────────────────────────────────────

def forward_vst(x: np.ndarray, a: float, b: float) -> np.ndarray:
    c = 0.375 + b / (a ** 2)
    val = np.maximum(0.0, x / a + c)
    return 2.0 * np.sqrt(val)

def inverse_vst_algebraic(f: np.ndarray, a: float, b: float) -> np.ndarray:
    c = 0.375 + b / (a ** 2)
    return a * ( (f ** 2) / 4.0 - c )

# ── 从 CSV 加载噪声参数 ────────────────────────────────────────────────────────

def load_noise_params(csv_path: str) -> dict[tuple[str, int, int], tuple[float, float]]:
    params = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['scene'], int(row['iso']), int(row['frame']))
            if key not in params:
                a_raw = float(row['a'])
                b_raw = float(row['b'])
                params[key] = (a_raw / RANGE, b_raw / RANGE ** 2)
    return params

# ── V-BM3D 时空块匹配 (多进程独立任务) ────────────────────────────────────────

def _match_chunk_vbm3d(
    coords_chunk: list[tuple[int, int, int]],
    guide_seq: np.ndarray,
    rad: int,
    temp_rad: int,
    patch_size: int,
    top_k: int,
) -> list[dict]:
    T, H, W = guide_seq.shape
    chunk_results = []

    for (t, ry, rx) in coords_chunk:
        # 参考块提取自当前帧 t
        ref = guide_seq[t, ry:ry + patch_size, rx:rx + patch_size]

        y0 = max(0, ry - rad);   y1 = min(H - patch_size, ry + rad)
        x0 = max(0, rx - rad);   x1 = min(W - patch_size, rx + rad)

        flat_d_list, flat_t_list, flat_y_list, flat_x_list = [], [], [], []

        # 时间窗口限制：跨帧搜索
        t_start = max(0, t - temp_rad)
        t_end   = min(T - 1, t + temp_rad)

        for tt in range(t_start, t_end + 1):
            region = guide_seq[tt, y0:y1 + patch_size, x0:x1 + patch_size]
            patches = sliding_window_view(region, (patch_size, patch_size))
            ny, nx = patches.shape[:2]

            dists = np.mean((patches - ref) ** 2, axis=(-2, -1))

            # 如果是在同帧内，剔除自己
            if tt == t:
                sy = ry - y0;  sx = rx - x0
                if 0 <= sy < ny and 0 <= sx < nx:
                    dists[sy, sx] = np.inf

            YY, XX = np.meshgrid(
                np.arange(y0, y0 + ny), np.arange(x0, x0 + nx), indexing='ij'
            )
            
            fd = dists.ravel()
            valid = np.isfinite(fd)
            
            flat_d_list.append(fd[valid])
            flat_t_list.append(np.full(np.count_nonzero(valid), tt, dtype=int))
            flat_y_list.append(YY.ravel()[valid])
            flat_x_list.append(XX.ravel()[valid])

        # 合并当前时间窗口内所有候选块，进行全局排序
        flat_d = np.concatenate(flat_d_list)
        flat_t = np.concatenate(flat_t_list)
        flat_y = np.concatenate(flat_y_list)
        flat_x = np.concatenate(flat_x_list)

        k = min(top_k, len(flat_d))
        if k == 0:
            chunk_results.append({'ref_pos': (t, ry, rx), 'top_matches': []})
            continue

        idx = np.argpartition(flat_d, k - 1)[:k]
        idx = idx[np.argsort(flat_d[idx])]
        top = [(float(flat_d[i]), int(flat_t[i]), int(flat_y[i]), int(flat_x[i])) for i in idx]
        chunk_results.append({'ref_pos': (t, ry, rx), 'top_matches': top})

    return chunk_results

def vectorized_match_parallel_vbm3d(guide_seq: np.ndarray, num_workers: int = NUM_WORKERS) -> list[dict]:
    T, H, W = guide_seq.shape
    rad = WIN // 2

    # 坐标现在包含时间维度 T
    coords = [
        (t, ry, rx)
        for t in range(T)
        for ry in range(0, H - PATCH + 1, STRIDE)
        for rx in range(0, W - PATCH + 1, STRIDE)
    ]

    chunk_size = max(1, len(coords) // num_workers)
    chunks = [coords[i:i + chunk_size] for i in range(0, len(coords), chunk_size)]

    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_match_chunk_vbm3d, chunk, guide_seq, rad, TEMP_RAD, PATCH, TOP_K)
            for chunk in chunks
        ]
        for fut in futures:
            all_results.extend(fut.result())

    return all_results

# ── V-BM3D 3D 协同滤波 ────────────────────────────────────────────────────────

def vbm3d_awgn_filtering(img_vst_seq: np.ndarray, all_results: list[dict]) -> np.ndarray:
    T, H, W = img_vst_seq.shape
    
    # 聚合数组扩展为序列级
    num_seq = np.zeros((T, H, W), dtype=np.float64)
    den_seq = np.zeros((T, H, W), dtype=np.float64)

    sigma_vst = 1.0
    threshold = 2.7 * sigma_vst

    for res in all_results:
        t_ref, ry, rx = res['ref_pos']
        # 坐标格式：(t, y, x)
        coords = [(t_ref, ry, rx)] + [(tt, y, x) for _, tt, y, x in res['top_matches']]

        group = np.stack(
            [img_vst_seq[tt, y:y + PATCH, x:x + PATCH] for tt, y, x in coords], axis=0
        ).astype(np.float64)

        # 3D DCT 变换
        freq = dctn(group, norm='ortho')
        
        freq[np.abs(freq) < threshold] = 0.0
        n_nz = int(np.count_nonzero(freq))
        
        w = 1.0 / (n_nz * (sigma_vst ** 2)) if n_nz > 0 else 1.0 / (sigma_vst ** 2)

        # 3D 逆变换
        denoised = idctn(freq, norm='ortho')
        
        # 将降噪后的块累加回其对应的时间帧和空间位置
        for i, (tt, y, x) in enumerate(coords):
            num_seq[tt, y:y + PATCH, x:x + PATCH] += denoised[i] * w
            den_seq[tt, y:y + PATCH, x:x + PATCH] += w

    out_seq = img_vst_seq.copy()
    mask = den_seq > 0
    out_seq[mask] = num_seq[mask] / den_seq[mask]
    return out_seq

# ── 视频序列级处理流程 ─────────────────────────────────────────────────────────

def process_sequence(args: tuple) -> list[dict]:
    scene, iso_val, nidx, frames, out_dir = args
    T = len(frames)

    # 1. 加载所有帧并进行前向 VST (考虑每帧不同的 a, b 参数)
    noisy_seq = []
    clean_seq = []
    noisy_vst_seq = []
    
    for f in frames:
        noisy_raw = cv2.imread(f['noisy_path'], cv2.IMREAD_UNCHANGED).astype(np.float32)
        clean_raw = cv2.imread(f['clean_path'], cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        n_img = np.clip((noisy_raw - BLACK) / RANGE, 0.0, 1.0)
        c_img = np.clip((clean_raw - BLACK) / RANGE, 0.0, 1.0)
        
        noisy_seq.append(n_img)
        clean_seq.append(c_img)
        noisy_vst_seq.append(forward_vst(n_img, f['a_norm'], f['sigma_norm']))

    noisy_seq = np.array(noisy_seq)
    clean_seq = np.array(clean_seq)
    noisy_vst_seq = np.array(noisy_vst_seq)

    # 生成序列级引导图 (逐帧空间高斯平滑)
    guide_seq = np.array([cv2.GaussianBlur(img, (5, 5), 1.5) for img in noisy_vst_seq])

    # 2. V-BM3D 时空块匹配
    t0 = time.perf_counter()
    matches = vectorized_match_parallel_vbm3d(guide_seq, num_workers=NUM_WORKERS)
    t_match = time.perf_counter() - t0

    # 3. V-BM3D 序列协同滤波
    t1 = time.perf_counter()
    denoised_vst_seq = vbm3d_awgn_filtering(noisy_vst_seq, matches)
    t_filter = time.perf_counter() - t1

    # 4. 后向逆 VST 并计算指标
    os.makedirs(out_dir, exist_ok=True)
    results = []
    
    for i, f in enumerate(frames):
        denoised = inverse_vst_algebraic(denoised_vst_seq[i], f['a_norm'], f['sigma_norm'])
        denoised = np.clip(denoised, 0.0, 1.0)

        psnr_in = float(skpsnr(clean_seq[i], noisy_seq[i], data_range=1.0))
        psnr_out = float(skpsnr(clean_seq[i], denoised, data_range=1.0))

        denoised_raw = np.clip(denoised * RANGE + BLACK, 0.0, WHITE).astype(np.uint16)
        out_tiff = os.path.join(out_dir, f"frame{f['fid']}_noisy{nidx}_denoised.tiff")
        cv2.imwrite(out_tiff, denoised_raw)

        results.append({
            'scene':            scene,
            'iso':              iso_val,
            'frame':            f['fid'],
            'noisy_idx':        nidx,
            'psnr_noisy_dB':    round(psnr_in,    4),
            'psnr_denoised_dB': round(psnr_out,   4),
            'time_match_s':     round(t_match / T, 1),  # 均摊到每帧的耗时
            'time_filter_s':    round(t_filter / T, 1),
            'a_norm':           round(f['a_norm'],      8),
            'sigma_norm':       round(f['sigma_norm'], 12),
        })

    return results

# ── 任务收集 (按序列打包) ──────────────────────────────────────────────────────

def collect_tasks(crvd_root, results_root, noise_params, iso_filter, scene_filter, noisy_filter):
    # 将属于同一个场景、ISO、以及同一个噪声分布索引的帧打包在一起
    seqs = defaultdict(list)
    
    for scene_dir in sorted(glob.glob(os.path.join(crvd_root, 'scene*'))):
        scene = os.path.basename(scene_dir)
        if scene_filter and scene != scene_filter: continue
            
        for iso_dir in sorted(glob.glob(os.path.join(scene_dir, 'ISO*'))):
            iso_name = os.path.basename(iso_dir)
            iso_val  = int(re.sub(r'\D', '', iso_name))
            if iso_filter and iso_val != iso_filter: continue
                
            out_dir = os.path.join(results_root, scene, iso_name)
            
            for clean_path in sorted(glob.glob(os.path.join(iso_dir, 'frame*_clean.tiff'))):
                m = re.search(r'frame(\d+)_clean\.tiff', os.path.basename(clean_path))
                if not m: continue
                fid = int(m.group(1))
                key = (scene, iso_val, fid)
                
                if key not in noise_params: continue
                a_norm, sigma_norm = noise_params[key]
                
                for np_path in sorted(glob.glob(os.path.join(iso_dir, f'frame{fid}_noisy*.tiff'))):
                    nm = re.search(r'frame\d+_noisy(\d+)\.tiff', os.path.basename(np_path))
                    nidx = int(nm.group(1)) if nm else -1
                    if noisy_filter is not None and nidx != noisy_filter: continue
                        
                    seqs[(scene, iso_val, nidx, out_dir)].append({
                        'noisy_path': np_path,
                        'clean_path': clean_path,
                        'fid': fid,
                        'a_norm': a_norm,
                        'sigma_norm': sigma_norm
                    })

    tasks = []
    for (scene, iso_val, nidx, out_dir), frames in seqs.items():
        # 确保时序正确
        frames.sort(key=lambda x: x['fid'])
        tasks.append((scene, iso_val, nidx, frames, out_dir))
        
    return tasks

# ── 主流程 ─────────────────────────────────────────────────────────────────────

def run(root, out, csv_out, params, iso, scene, noisy):
    noise_params = load_noise_params(params)
    tasks = collect_tasks(root, out, noise_params, iso, scene, noisy)
    
    print(f"共发现 {len(tasks)} 个视频序列。内部 40 核并行处理跨帧块匹配…\n")

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    rows = []

    pbar = tqdm(tasks, desc='V-BM3D-CRVD', dynamic_ncols=True)
    for task in pbar:
        try:
            seq_results = process_sequence(task)
            rows.extend(seq_results)
            # 取序列的平均 PSNR 用于显示
            avg_psnr = np.mean([r['psnr_denoised_dB'] for r in seq_results])
            pbar.set_postfix({
                'ISO': task[1],
                'Seq': f"n{task[2]}",
                'AvgPSNR↑': f"{avg_psnr:.2f} dB",
            })
        except Exception as e:
            tqdm.write(f"[ERR] Sequence {task[0]} ISO{task[1]}: {e}")
    pbar.close()

    rows.sort(key=lambda r: (r['scene'], r['iso'], r['noisy_idx'], r['frame']))

    fieldnames = ['scene', 'iso', 'frame', 'noisy_idx', 'psnr_noisy_dB', 'psnr_denoised_dB', 'time_match_s', 'time_filter_s', 'a_norm', 'sigma_norm']
    with open(csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',   default='data/CRVD/noisy')
    parser.add_argument('--out',    default='out/results/CRVD/my_vbm3d_vst')
    parser.add_argument('--csv',    default='out/results/CRVD/my_vbm3d_vst/results.csv')
    parser.add_argument('--params', default='out/results/crvd_noise_params.csv')
    parser.add_argument('--iso', type=int, default=None)
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--noisy', type=int, default=None)
    args = parser.parse_args()

    run(args.root, args.out, args.csv, args.params, args.iso, args.scene, args.noisy)