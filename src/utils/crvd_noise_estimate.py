"""
CRVD 数据集噪声参数估计工具

对每个 (scene, ISO, frame) 组合：
  - 利用 10 个 noisy 帧（j=0..9）估计泊松-高斯噪声参数 a, b
      Var(noisy | clean) = a * (clean - BLACK) + b
  - 计算每个 noisy 帧与 clean 帧之间的 PSNR（RAW 域，uint16）

输出：CSV 文件，每行对应一个 (scene, ISO, frame, noisy_idx)。
"""

import os
import glob
import csv
import re
import cv2
import numpy as np

# ── 相机参数（与 crvd_raw2png.py 保持一致） ──────────────────────────────────
BLACK_LEVEL = 240.0
WHITE_LEVEL = 4095.0
MAX_VAL = WHITE_LEVEL  # PSNR 计算用的峰值


# ── 基础工具 ──────────────────────────────────────────────────────────────────

def read_raw(path: str) -> np.ndarray:
    """读取单通道 uint16 RAW TIFF，返回 float32 数组。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取: {path}")
    return img.astype(np.float32)


def psnr(noisy: np.ndarray, clean: np.ndarray, max_val: float = MAX_VAL) -> float:
    """在 RAW uint16 域计算 PSNR（dB）。"""
    mse = np.mean((noisy - clean) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(max_val / np.sqrt(mse))


def estimate_noise_params(noisy_stack: np.ndarray) -> tuple[float, float]:
    """
    用多帧 noisy 估计泊松-高斯噪声参数 a, b。

    参数
    ----
    noisy_stack : shape (N, H, W), float32，N 个 noisy 帧（原始 uint16 值）

    返回
    ----
    a : 泊松分量系数（shot noise）
    b : 高斯分量方差（read noise variance），单位与 (raw - BLACK)^2 一致

    原理
    ----
    令 x = noisy - BLACK（归零后的信号）
    对每个像素 p，N 帧 x_p 的:
        mean_p  ≈ E[clean_p - BLACK]
        var_p   ≈ a * mean_p + b
    用 OLS 拟合 var ~ mean 得到 a, b。
    为减少离群点影响，仅保留 mean 落在 [q5, q95] 范围内的像素。
    """
    x = noisy_stack - BLACK_LEVEL                    # 黑电平归零
    pixel_mean = x.mean(axis=0).ravel()              # (H*W,)
    pixel_var  = x.var(axis=0, ddof=1).ravel()       # (H*W,)

    # 过滤：只保留信号在合理范围内的像素
    lo, hi = np.percentile(pixel_mean, [5, 95])
    mask = (pixel_mean >= max(lo, 0)) & (pixel_mean <= hi) & (pixel_var >= 0)
    m = pixel_mean[mask]
    v = pixel_var[mask]

    if len(m) < 100:
        return float('nan'), float('nan')

    # OLS：var = a * mean + b  →  [mean, 1] @ [a, b]^T
    A = np.stack([m, np.ones_like(m)], axis=1)
    result = np.linalg.lstsq(A, v, rcond=None)
    a, b = result[0]
    return float(a), float(b)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def process_crvd(
    crvd_root: str = "data/CRVD/noisy",
    output_csv: str = "out/results/crvd_noise_params.csv",
) -> None:
    """
    遍历 crvd_root 下所有 scene/ISO/frame 组合，
    估计噪声参数并计算 PSNR，结果写入 output_csv。
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    rows = []

    scene_dirs = sorted(glob.glob(os.path.join(crvd_root, "scene*")))
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)

        iso_dirs = sorted(glob.glob(os.path.join(scene_dir, "ISO*")))
        for iso_dir in iso_dirs:
            iso_name = os.path.basename(iso_dir)                     # e.g. "ISO1600"
            iso_val  = int(re.sub(r"[^0-9]", "", iso_name))          # e.g. 1600

            # 找出该目录下有哪些 frame 编号
            clean_paths = sorted(glob.glob(os.path.join(iso_dir, "frame*_clean.tiff")))
            frame_ids = []
            for cp in clean_paths:
                m = re.search(r"frame(\d+)_clean\.tiff", os.path.basename(cp))
                if m:
                    frame_ids.append(int(m.group(1)))

            for frame_id in frame_ids:
                clean_path = os.path.join(iso_dir, f"frame{frame_id}_clean.tiff")
                noisy_paths = sorted(glob.glob(
                    os.path.join(iso_dir, f"frame{frame_id}_noisy*.tiff")
                ))
                if not noisy_paths:
                    continue

                # 读取 clean
                try:
                    clean_img = read_raw(clean_path)
                except FileNotFoundError as e:
                    print(f"[WARN] {e}")
                    continue

                # 读取所有 noisy 帧
                noisy_stack = []
                noisy_indices = []
                for np_path in noisy_paths:
                    m = re.search(r"frame\d+_noisy(\d+)\.tiff", os.path.basename(np_path))
                    noisy_idx = int(m.group(1)) if m else -1
                    try:
                        noisy_stack.append(read_raw(np_path))
                        noisy_indices.append(noisy_idx)
                    except FileNotFoundError as e:
                        print(f"[WARN] {e}")

                if not noisy_stack:
                    continue

                stack_arr = np.stack(noisy_stack, axis=0)   # (N, H, W)

                # 估计噪声参数（利用全部 N 帧）
                a, b = estimate_noise_params(stack_arr)

                # 计算每帧 PSNR
                for idx, noisy_img in zip(noisy_indices, noisy_stack):
                    p = psnr(noisy_img, clean_img)
                    rows.append({
                        "scene":     scene_name,
                        "iso":       iso_val,
                        "frame":     frame_id,
                        "noisy_idx": idx,
                        "psnr_db":   round(p, 4),
                        "a":         round(a, 6) if not np.isnan(a) else "",
                        "b":         round(b, 4) if not np.isnan(b) else "",
                    })

                print(f"[OK] {scene_name}/{iso_name}/frame{frame_id}  "
                      f"a={a:.4f}  b={b:.2f}  "
                      f"PSNR(mean)={np.mean([r['psnr_db'] for r in rows if r['scene']==scene_name and r['iso']==iso_val and r['frame']==frame_id]):.2f} dB")

    # 写 CSV
    fieldnames = ["scene", "iso", "frame", "noisy_idx", "psnr_db", "a", "b"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n结果已保存至: {output_csv}  ({len(rows)} 行)")


# ── 汇总统计（可选） ──────────────────────────────────────────────────────────

def summarize(output_csv: str = "out/results/crvd_noise_params.csv") -> None:
    """按 ISO 打印噪声参数和 PSNR 的均值，方便快速核查。"""
    import csv as _csv
    from collections import defaultdict

    data = defaultdict(list)
    with open(output_csv) as f:
        for row in _csv.DictReader(f):
            iso = int(row["iso"])
            data[iso].append((float(row["psnr_db"]),
                              float(row["a"]) if row["a"] else float('nan'),
                              float(row["b"]) if row["b"] else float('nan')))

    print(f"\n{'ISO':>8}  {'PSNR(dB)':>10}  {'a':>10}  {'b':>12}  {'N':>6}")
    print("-" * 55)
    for iso in sorted(data):
        vals = np.array(data[iso])
        print(f"{iso:>8}  "
              f"{np.nanmean(vals[:,0]):>10.3f}  "
              f"{np.nanmean(vals[:,1]):>10.6f}  "
              f"{np.nanmean(vals[:,2]):>12.2f}  "
              f"{len(vals):>6}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRVD 泊松-高斯噪声参数估计 & PSNR 计算")
    parser.add_argument("--root",   default="data/CRVD/noisy",              help="CRVD noisy 根目录")
    parser.add_argument("--output", default="out/results/crvd_noise_params.csv", help="输出 CSV 路径")
    parser.add_argument("--summary", action="store_true",                   help="处理完毕后打印汇总统计")
    args = parser.parse_args()

    process_crvd(crvd_root=args.root, output_csv=args.output)

    if args.summary:
        summarize(args.output)
