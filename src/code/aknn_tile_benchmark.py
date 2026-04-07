"""
AKNN-BM3D Tile 尺寸 Benchmark
搜索范围 = core（stride）in RAW domain，halo（overlap）固定 = 19 packed（38 RAW）。

当前配置：core=128×128 RAW → stride=64 packed，tile=83×83 packed，overlap=19 packed。

测试配置（core in RAW → tile in packed = stride + 19）：
  64×64   → tile 51×51  packed
  128×72  → tile 83×55  packed  (非正方形)
  128×128 → tile 83×83  packed  (当前)
  256×256 → tile 147×147 packed
  512×512 → tile 275×275 packed

测试数据：scene1/ISO3200/noisy_idx=0/frame1（单帧，无时域），40 核并行。
"""

from __future__ import annotations
import os, time, csv
import concurrent.futures

import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.code.AKNN_BM3D_v8_raw import (
    CRVD_BLACK, CRVD_WHITE, CRVD_RANGE, CRVD_GR_INDEX,
    load_crvd_noise_params,
    pack_bayer, forward_gat, inverse_gat, unpack_bayer,
    process_single_block_video_tile_raw,
)
from skimage.metrics import peak_signal_noise_ratio as skpsnr

NUM_WORKERS = 40
FIXED_OVERLAP_PACKED = 19   # halo 固定不变


# ── 支持非正方形 Tile 的切分 ──────────────────────────────────────────────────

def split_rect(img: np.ndarray,
               tile_h: int, tile_w: int,
               overlap: int):
    """
    按矩形 Tile 切分 (H, W, ...) 图像。
    overlap 在 H、W 方向相同（halo 固定）。
    返回 (blocks, coords)，coords = [(y0, y1, x0, x1), ...]。
    """
    H, W = img.shape[:2]
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    coords, blocks = [], []

    for y in range(0, H, stride_h):
        for x in range(0, W, stride_w):
            y0, x0 = y, x
            y1 = min(H, y + tile_h)
            x1 = min(W, x + tile_w)
            # 边界对齐
            if y1 - y0 < tile_h and H >= tile_h:
                y0 = H - tile_h; y1 = H
            if x1 - x0 < tile_w and W >= tile_w:
                x0 = W - tile_w; x1 = W
            entry = (y0, y1, x0, x1)
            if entry not in coords:
                coords.append(entry)
                blocks.append(img[y0:y1, x0:x1].copy())

    return blocks, coords


# ── 单帧处理（参数化 Tile）────────────────────────────────────────────────────

def process_frame(noisy_path: str, clean_path: str,
                  a_gat: float, sigma_gat: float,
                  tile_h_packed: int, tile_w_packed: int,
                  K_spatial: int = 7, patch_size: int = 7, step: int = 2) -> dict:
    """
    用指定 packed-域 Tile 尺寸处理单帧（无时域），返回耗时和 PSNR。
    """
    noisy_raw = cv2.imread(noisy_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    clean_raw = cv2.imread(clean_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    noisy_norm = np.clip((noisy_raw - CRVD_BLACK) / CRVD_RANGE, 0.0, 1.0)
    clean_norm = np.clip((clean_raw - CRVD_BLACK) / CRVD_RANGE, 0.0, 1.0)
    psnr_noisy = float(skpsnr(clean_norm, noisy_norm, data_range=1.0))

    noisy_packed = pack_bayer(noisy_norm)
    curr_packed_vst = np.zeros_like(noisy_packed, dtype=np.float32)
    for ch in range(4):
        curr_packed_vst[..., ch] = forward_gat(noisy_packed[..., ch], a_gat, sigma_gat)

    curr_guide_vst = np.zeros_like(curr_packed_vst)
    for ch in range(4):
        curr_guide_vst[..., ch] = cv2.GaussianBlur(curr_packed_vst[..., ch], (5, 5), 1.5)

    curr_noisy_blocks, block_coords = split_rect(
        curr_packed_vst, tile_h_packed, tile_w_packed, FIXED_OVERLAP_PACKED)
    curr_guide_blocks, _ = split_rect(
        curr_guide_vst, tile_h_packed, tile_w_packed, FIXED_OVERLAP_PACKED)

    n_tiles = len(curr_noisy_blocks)
    denoised_vst_blocks = [None] * n_tiles

    t_start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as exe:
        futures = []
        for i in range(n_tiles):
            gr = curr_guide_blocks[i][..., CRVD_GR_INDEX]
            futures.append(exe.submit(
                process_single_block_video_tile_raw,
                i, curr_noisy_blocks[i], gr,
                None, None, K_spatial, 0, patch_size, step,
            ))
        for fut in concurrent.futures.as_completed(futures):
            idx, blk, _, _ = fut.result()
            denoised_vst_blocks[idx] = blk
    t_elapsed = time.perf_counter() - t_start

    H_pack, W_pack = curr_packed_vst.shape[:2]
    num = np.zeros((H_pack, W_pack, 4), dtype=np.float64)
    den = np.zeros((H_pack, W_pack, 4), dtype=np.float64)
    for i, (y0, y1, x0, x1) in enumerate(block_coords):
        num[y0:y1, x0:x1, :] += denoised_vst_blocks[i]
        den[y0:y1, x0:x1, :] += 1.0
    denoised_packed_vst = (num / den).astype(np.float32)

    denoised_packed = np.zeros_like(noisy_packed)
    for ch in range(4):
        denoised_packed[..., ch] = np.clip(
            inverse_gat(denoised_packed_vst[..., ch], a_gat, sigma_gat), 0.0, 1.0
        ).astype(np.float32)
    denoised_norm = unpack_bayer(denoised_packed)
    psnr_out = float(skpsnr(clean_norm, denoised_norm, data_range=1.0))

    return {
        'n_tiles':       n_tiles,
        'time_s':        round(t_elapsed, 2),
        'psnr_noisy':    round(psnr_noisy, 3),
        'psnr_denoised': round(psnr_out,   3),
    }


# ── 主程序 ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    NOISY = 'data/CRVD/noisy/scene1/ISO3200/frame1_noisy0_720p.tiff'
    CLEAN = 'data/CRVD/noisy/scene1/ISO3200/frame1_clean_720p.tiff'

    params = load_crvd_noise_params('out/results/crvd_noise_params.csv')
    a_gat, sigma_gat = params[('scene1', 3200, 1)]

    # (name, core_h_raw, core_w_raw)
    # tile_packed = core/2 + FIXED_OVERLAP_PACKED(19)
    configs = [
        ('64×64',    64,   64),
        ('128×72',  128,   72),
        ('72x128',  72,   128),
        ('128×128', 128,  128),   # 当前
        ('256×256', 256,  256),
        ('512×512', 512,  512),
        ('256x144', 256,  144), 
        ('144x256', 144,  256),   # 非正方形
    ]

    print(f"固定 halo = {FIXED_OVERLAP_PACKED} packed（{FIXED_OVERLAP_PACKED*2} RAW）\n")
    print(f"{'配置':>10}  {'tile(packed)':>13}  {'N tiles':>8}  "
          f"{'耗时(s)':>9}  {'PSNR_in':>8}  {'PSNR_out':>9}  {'增益':>7}")
    print('-' * 75)

    results = []
    for name, ch_raw, cw_raw in configs:
        th = ch_raw // 2 + FIXED_OVERLAP_PACKED   # tile_h packed
        tw = cw_raw // 2 + FIXED_OVERLAP_PACKED   # tile_w packed
        print(f"  测试 {name}  (tile {th}×{tw} packed)...", flush=True)

        r = process_frame(NOISY, CLEAN, a_gat, sigma_gat, th, tw)
        gain = r['psnr_denoised'] - r['psnr_noisy']
        tile_str = f"{th}×{tw}"
        print(f"{name:>10}  {tile_str:>13}  {r['n_tiles']:>8}  "
              f"{r['time_s']:>9.2f}  {r['psnr_noisy']:>8.3f}  "
              f"{r['psnr_denoised']:>9.3f}  {gain:>7.3f}")

        results.append({
            'config': name,
            'tile_packed': tile_str,
            **r,
            'psnr_gain': round(gain, 3),
        })

    fastest = min(results, key=lambda x: x['time_s'])
    print(f"\n最快: {fastest['config']}  →  {fastest['time_s']}s  "
          f"({fastest['n_tiles']} tiles)")

    os.makedirs('out/results', exist_ok=True)
    csv_out = 'out/results/aknn_tile_benchmark.csv'
    with open(csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'config', 'tile_packed', 'n_tiles', 'time_s',
            'psnr_noisy', 'psnr_denoised', 'psnr_gain'])
        w.writeheader()
        w.writerows(results)
    print(f"结果已保存：{csv_out}")
