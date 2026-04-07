#!/usr/bin/env python3
"""
运行所有 7 种降噪方法，仅处理 CRVD scene1 中 noisy_idx=0 的图像，
合并结果输出到 out/crvd_scene1_noisy0_all.csv。

列：method, scene, iso, frame, noisy_idx,
    psnr_noisy_dB, psnr_denoised_dB, time_all, a_norm, sigma_norm
"""
import sys, os, glob, re, csv, importlib.util

# ── 将 src/code 加入搜索路径 ─────────────────────────────────────────────────
REPO   = os.path.dirname(os.path.abspath(__file__))
SRCDIR = os.path.join(REPO, 'src', 'code')
sys.path.insert(0, SRCDIR)

CRVD_ROOT  = os.path.join(REPO, 'data', 'CRVD', 'noisy')
PARAMS_CSV = os.path.join(REPO, 'out', 'results', 'crvd_noise_params.csv')
OUT_ROOT   = os.path.join(REPO, 'out')
OUT_CSV    = os.path.join(OUT_ROOT, 'crvd_scene1_noisy0_all.csv')
SCENE      = 'scene1'
NOISY_IDX  = 0
BLACK      = 240.0
WHITE      = 4095.0
RANGE      = WHITE - BLACK   # 3855.0


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def load_module(name: str) -> object:
    path = os.path.join(SRCDIR, f'{name}.py')
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod      # 注册后 ProcessPoolExecutor pickle 才能找到
    spec.loader.exec_module(mod)
    return mod


def load_noise_params_generic():
    """返回 {(scene, iso, frame): (a_norm, sigma_norm)}"""
    params = {}
    with open(PARAMS_CSV, newline='') as f:
        for row in csv.DictReader(f):
            key = (row['scene'], int(row['iso']), int(row['frame']))
            if key not in params:
                a_raw = float(row['a'])
                b_raw = max(float(row['b']), 0.0)
                params[key] = (a_raw / RANGE, b_raw / RANGE ** 2)
    return params


def collect_tasks(results_root: str):
    """收集 scene1/noisy0 的所有 (task_tuple, iso_dir, iso_val, fid) 信息。"""
    noise_params = load_noise_params_generic()
    tasks = []
    scene_dir = os.path.join(CRVD_ROOT, SCENE)
    for iso_dir in sorted(glob.glob(os.path.join(scene_dir, 'ISO*'))):
        iso_name = os.path.basename(iso_dir)
        iso_val  = int(re.sub(r'\D', '', iso_name))
        out_dir  = os.path.join(results_root, SCENE, iso_name)
        for clean_path in sorted(glob.glob(os.path.join(iso_dir, 'frame*_clean.tiff'))):
            m = re.search(r'frame(\d+)_clean\.tiff', os.path.basename(clean_path))
            if not m:
                continue
            fid = int(m.group(1))
            key = (SCENE, iso_val, fid)
            if key not in noise_params:
                continue
            a_norm, sigma_norm = noise_params[key]
            noisy_path = os.path.join(iso_dir, f'frame{fid}_noisy{NOISY_IDX}.tiff')
            if not os.path.exists(noisy_path):
                continue
            tasks.append({
                'tuple':     (noisy_path, clean_path, iso_val, SCENE, fid,
                              NOISY_IDX, out_dir, a_norm, sigma_norm),
                'iso_dir':   iso_dir,
                'iso_val':   iso_val,
                'fid':       fid,
                'a_norm':    a_norm,
                'sigma_norm': sigma_norm,
            })
    return tasks


def normalize_row(row: dict, method: str) -> dict:
    """将各方法输出的行统一化为目标列格式。"""
    # time_all：各类时间字段求和
    if 'time_match_s' in row and 'time_filter_s' in row:
        t = row['time_match_s'] + row['time_filter_s']
    elif 'time_s' in row:
        t = row['time_s']
    elif 'time_filter_s' in row:
        t = row['time_filter_s']
    else:
        t = 0.0

    # a_norm：各方法字段名不同
    a = row.get('a_norm') or row.get('a_gat') or row.get('beta1') or 0.0

    # sigma_norm（方差 = b/RANGE²）：
    #   my_brute_bm3d*, bilateral, nlm  → 'sigma_norm'  = b/RANGE²
    #   bm3d_baseline_RAW               → 'beta2'        = b/RANGE²
    #   AKNN                            → 'sigma_gat'    = sqrt(b)/RANGE
    #                                     → sigma_gat² = b/RANGE² = sigma_norm
    if 'sigma_norm' in row:
        s = row['sigma_norm']
    elif 'beta2' in row:
        s = row['beta2']
    elif 'sigma_gat' in row:
        s = row['sigma_gat'] ** 2
    else:
        s = 0.0

    return {
        'method':           method,
        'scene':            row['scene'],
        'iso':              row['iso'],
        'frame':            row['frame'],
        'noisy_idx':        row['noisy_idx'],
        'psnr_noisy_dB':    row['psnr_noisy_dB'],
        'psnr_denoised_dB': row['psnr_denoised_dB'],
        'time_all':         round(t, 3),
        'a_norm':           round(float(a), 8),
        'sigma_norm':       round(float(s), 12),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 各方法运行器
# ─────────────────────────────────────────────────────────────────────────────

def run_per_image(method_name: str, module_func_name: str = 'process_one') -> list[dict]:
    """适用于所有按单图处理的方法（brute_bm3d_no_VST、brute_bm3d、bilateral、nlm）。"""
    print(f'\n{"="*60}')
    print(f'[{method_name}] 开始...')
    mod    = load_module(method_name)
    func   = getattr(mod, module_func_name)
    res_dir = os.path.join(OUT_ROOT, 'tmp', method_name)
    tasks  = collect_tasks(res_dir)
    rows   = []
    for t in tasks:
        try:
            r = func(t['tuple'])
            rows.append(normalize_row(r, method_name))
            print(f"  ISO{t['iso_val']} frame{t['fid']} → "
                  f"PSNR {rows[-1]['psnr_denoised_dB']:.2f} dB")
        except Exception as e:
            print(f"  [ERR] {t['tuple'][0]}: {e}")
    print(f'[{method_name}] 完成，共 {len(rows)} 行。')
    return rows


def run_bm3d_baseline_raw() -> list[dict]:
    """bm3d_baseline_RAW: process_one_image"""
    method = 'bm3d_baseline_RAW'
    print(f'\n{"="*60}')
    print(f'[{method}] 开始...')
    mod    = load_module(method)
    res_dir = os.path.join(OUT_ROOT, 'tmp', method)
    tasks  = collect_tasks(res_dir)
    rows   = []
    for t in tasks:
        try:
            r = mod.process_one_image(t['tuple'])
            rows.append(normalize_row(r, method))
            print(f"  ISO{t['iso_val']} frame{t['fid']} → "
                  f"PSNR {rows[-1]['psnr_denoised_dB']:.2f} dB")
        except Exception as e:
            print(f"  [ERR] {t['tuple'][0]}: {e}")
    print(f'[{method}] 完成，共 {len(rows)} 行。')
    return rows


def run_vbm3d() -> list[dict]:
    """my_brute_vbm3d: process_sequence 处理整个序列。"""
    method = 'my_brute_vbm3d'
    print(f'\n{"="*60}')
    print(f'[{method}] 开始...')
    mod    = load_module(method)
    noise_params = load_noise_params_generic()
    scene_dir = os.path.join(CRVD_ROOT, SCENE)
    rows = []

    for iso_dir in sorted(glob.glob(os.path.join(scene_dir, 'ISO*'))):
        iso_name = os.path.basename(iso_dir)
        iso_val  = int(re.sub(r'\D', '', iso_name))
        out_dir  = os.path.join(OUT_ROOT, 'tmp', method, SCENE, iso_name)

        # 收集该 ISO 下所有帧，打包为 frames 列表
        frames = []
        for clean_path in sorted(glob.glob(os.path.join(iso_dir, 'frame*_clean.tiff'))):
            m = re.search(r'frame(\d+)_clean\.tiff', os.path.basename(clean_path))
            if not m:
                continue
            fid = int(m.group(1))
            key = (SCENE, iso_val, fid)
            if key not in noise_params:
                continue
            noisy_path = os.path.join(iso_dir, f'frame{fid}_noisy{NOISY_IDX}.tiff')
            if not os.path.exists(noisy_path):
                continue
            a_norm, sigma_norm = noise_params[key]
            frames.append({
                'noisy_path': noisy_path,
                'clean_path': clean_path,
                'fid':        fid,
                'a_norm':     a_norm,
                'sigma_norm': sigma_norm,
            })

        if not frames:
            continue

        frames.sort(key=lambda x: x['fid'])
        args = (SCENE, iso_val, NOISY_IDX, frames, out_dir)
        try:
            seq_rows = mod.process_sequence(args)
            for r in seq_rows:
                rows.append(normalize_row(r, method))
                print(f"  ISO{iso_val} frame{r['frame']} → "
                      f"PSNR {rows[-1]['psnr_denoised_dB']:.2f} dB")
        except Exception as e:
            print(f"  [ERR] {SCENE}/{iso_name}: {e}")

    print(f'[{method}] 完成，共 {len(rows)} 行。')
    return rows


def run_aknn() -> list[dict]:
    """AKNN_BM3D_v8_raw: process_crvd_sequence，逐 ISO 目录调用。"""
    method = 'AKNN_BM3D_v8_raw'
    print(f'\n{"="*60}')
    print(f'[{method}] 开始...')
    mod = load_module(method)

    # AKNN 有自己的 load_crvd_noise_params（sigma_gat 格式）
    noise_params = mod.load_crvd_noise_params(PARAMS_CSV)
    scene_dir = os.path.join(CRVD_ROOT, SCENE)
    rows = []

    for iso_dir in sorted(glob.glob(os.path.join(scene_dir, 'ISO*'))):
        iso_name = os.path.basename(iso_dir)
        iso_val  = int(re.sub(r'\D', '', iso_name))
        out_dir  = os.path.join(OUT_ROOT, 'tmp', method, SCENE, iso_name)
        try:
            seq_rows = mod.process_crvd_sequence(
                iso_dir      = iso_dir,
                noisy_idx    = NOISY_IDX,
                scene        = SCENE,
                iso_val      = iso_val,
                noise_params = noise_params,
                out_dir      = out_dir,
            )
            for r in seq_rows:
                rows.append(normalize_row(r, method))
                print(f"  ISO{iso_val} frame{r['frame']} → "
                      f"PSNR {rows[-1]['psnr_denoised_dB']:.2f} dB")
        except Exception as e:
            print(f"  [ERR] {SCENE}/{iso_name}: {e}")

    print(f'[{method}] 完成，共 {len(rows)} 行。')
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    all_rows: list[dict] = []

    # 1. my_brute_bm3d_no_VST（无 VST，自适应泊松-高斯阈值）
    all_rows.extend(run_per_image('my_brute_bm3d_no_VST'))

    # 2. my_brute_bm3d（含 VST）
    all_rows.extend(run_per_image('my_brute_bm3d'))

    # 3. my_brute_vbm3d（时空 V-BM3D）
    all_rows.extend(run_vbm3d())

    # 4. baseline_bilateral_vst
    all_rows.extend(run_per_image('baseline_bilateral_vst'))

    # 5. baseline_nlm_vst
    all_rows.extend(run_per_image('baseline_nlm_vst'))

    # 6. bm3d_baseline_RAW（官方 bm3d 库 + VST）
    all_rows.extend(run_bm3d_baseline_raw())

    # 7. AKNN_BM3D_v8_raw
    all_rows.extend(run_aknn())

    # ── 排序 ──
    all_rows.sort(key=lambda r: (r['method'], r['scene'], r['iso'],
                                  r['frame'], r['noisy_idx']))

    # ── 写 CSV ──
    fieldnames = [
        'method', 'scene', 'iso', 'frame', 'noisy_idx',
        'psnr_noisy_dB', 'psnr_denoised_dB', 'time_all',
        'a_norm', 'sigma_norm',
    ]
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f'\n{"="*60}')
    print(f'全部完成！结果已保存至 {OUT_CSV}  （共 {len(all_rows)} 行）')
    print()

    # ── 简要汇总 ──
    from collections import defaultdict
    import numpy as np
    stats = defaultdict(list)
    for r in all_rows:
        stats[r['method']].append(r['psnr_denoised_dB'])

    print(f"{'Method':<30}  {'avg PSNR_denoised(dB)':>22}  {'N':>4}")
    print('-' * 62)
    for method in sorted(stats):
        arr = np.array(stats[method])
        print(f"{method:<30}  {np.mean(arr):>22.3f}  {len(arr):>4}")


if __name__ == '__main__':
    main()
