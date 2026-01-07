import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim



def log_print(msg: str):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def read_gray_float01(path: Path) -> np.ndarray:
    img_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img_u8.astype(np.float32) / 255.0

def save_gray01_png(path: Path, img01: np.ndarray):
    """保存 [0,1] 灰度图到 png（uint8）"""
    img_u8 = np.clip(img01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img_u8)

# =========================
# Step 1 - CONFIG (只改这里)
# =========================
base = Path("../Datasets/DAVIS/480p/bus-Y")

# GT（干净中间帧）
gt_path = base / "ori_photo" / "1_Y.png"

# noisy 三帧（0/1/2）
noisy_dir = base / "ori_photo_20AWGN_123456"
tm1_path = noisy_dir / "y_00_noise.png"
t_path   = noisy_dir / "y_01_noise.png"
tp1_path = noisy_dir / "y_02_noise.png"

# 噪声强度（必须和加噪一致）
sigma_255 = 20
sigma = sigma_255 / 255.0

# 输出目录（本 step 只做拷贝保存 + log）
out_dir = base / f"vbm3d_step1_{sigma_255}AWGN_123456"
out_dir.mkdir(parents=True, exist_ok=True)

log_path = out_dir / "run_log.txt"

log_print("\n" + "=" * 70)
log_print(f"[RUN] {time.strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"[CFG] base      = {base}")
log_print(f"[CFG] sigma_255 = {sigma_255}, sigma={sigma:.6f}")
log_print(f"[CFG] out_dir   = {out_dir}")


# =========================
# Step 2 - PARAMS（匹配与组块参数）
# =========================
patch_size = 8          # patch 边长（BM3D常用 8）
stride = 3              # patch 网格步长（越小越密越慢）
search_radius = 12      # 搜索窗口半径（像素），窗口边长 = 2R+1
top_k = 16              # Top-K 相似块数（VBM3D 常取 16/32/64）
border_mode = "reflect" # padding 边界模式：reflect/edge/constant

# Debug：为了不爆算力，先只保存/处理部分 patch 的组块（你确认没问题再放开）
max_patches_debug = 30  # 只对前 N 个 reference patch 做匹配与保存（先跑通）
save_match_vis = True   # 是否保存“匹配位置示意图”（帮助确认搜索窗口正确）

dbg_blocks_dir = out_dir / "dbg_blocks"
dbg_blocks_dir.mkdir(parents=True, exist_ok=True)

dbg_vis_dir = out_dir / "dbg_match_vis"
dbg_vis_dir.mkdir(parents=True, exist_ok=True)


def pad_image(img: np.ndarray, pad: int, mode: str) -> np.ndarray:
    """给图像四周 padding，避免 patch 越界"""
    if pad <= 0:
        return img
    if mode == "reflect":
        return np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    if mode == "edge":
        return np.pad(img, ((pad, pad), (pad, pad)), mode="edge")
    if mode == "constant":
        return np.pad(img, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    raise ValueError(f"Unknown border_mode: {mode}")

def extract_patch(padded_img: np.ndarray, y: int, x: int, p: int) -> np.ndarray:
    """
    从 padded_img 上取 patch。
    注意：这里的 (y,x) 是“原图坐标”，因为 padded_img 已经整体 pad 了 pad 个像素，
         所以实际取 patch 的起点要 +pad，但我们在调用时会统一管理。
    """
    return padded_img[y:y+p, x:x+p]


def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    """L2 距离（平方和也行，这里用均方更稳一点）"""
    diff = a - b
    return float(np.mean(diff * diff))


def make_collage(patches: np.ndarray, cols: int = 8, scale: int = 6) -> np.ndarray:
    """
    把 (K, p, p) 的 patch 做成拼图可视化，便于检查匹配效果。
    - cols: 每行多少个
    - scale: 放大倍数（方便看）
    """
    K, p, _ = patches.shape
    rows = int(np.ceil(K / cols))
    canvas = np.zeros((rows * p, cols * p), dtype=np.float32)
    for i in range(K):
        r = i // cols
        c = i % cols
        canvas[r*p:(r+1)*p, c*p:(c+1)*p] = patches[i]
    if scale > 1:
        canvas = cv2.resize(canvas, (canvas.shape[1]*scale, canvas.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    return canvas

def draw_search_and_matches(img01: np.ndarray, ref_xy, matches_xy, R: int, p: int) -> np.ndarray:
    """
    在图上画：reference patch、搜索窗口、Top-K 匹配位置。
    只用于 debug，帮助确认你“搜索窗口是否围绕 ref 坐标”。
    """
    img_u8 = np.clip(img01 * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    y, x = ref_xy

    # reference patch 框（绿）
    cv2.rectangle(vis, (x, y), (x + p - 1, y + p - 1), (0, 255, 0), 1)

    # search window 框（蓝）
    y0 = max(0, y - R)
    x0 = max(0, x - R)
    y1 = min(img01.shape[0] - p, y + R)
    x1 = min(img01.shape[1] - p, x + R)
    cv2.rectangle(vis, (x0, y0), (x1 + p - 1, y1 + p - 1), (255, 0, 0), 1)

    # matches（红）
    for (my, mx) in matches_xy:
        cv2.rectangle(vis, (mx, my), (mx + p - 1, my + p - 1), (0, 0, 255), 1)

    return vis


# =========================
# Step 2 - LOAD
# =========================
gt = read_gray_float01(gt_path)
Itm1 = read_gray_float01(tm1_path)
It   = read_gray_float01(t_path)
Itp1 = read_gray_float01(tp1_path)

H, W = It.shape
log_print(f"[IMG] shape HxW = {H} x {W}")
log_print(f"[IMG] paths: tm1={tm1_path.name}, t={t_path.name}, tp1={tp1_path.name}")

# 记录一下 noisy vs GT 的 baseline（只是 log，不影响算法）
ssim_noisy = ssim(gt, It, data_range=1.0)
mse_noisy = float(np.mean((gt - It) ** 2))
log_print(f"[BASELINE] noisy center: SSIM={ssim_noisy:.6f}, MSE={mse_noisy:.8f}")

# 可选：保存输入图，确认读对了
save_gray01_png(out_dir / "noisy_tm1.png", Itm1)
save_gray01_png(out_dir / "noisy_t.png", It)
save_gray01_png(out_dir / "noisy_tp1.png", Itp1)
save_gray01_png(out_dir / "gt.png", gt)

# =========================
# Step 2 - CORE（block matching + 3D stack）
# =========================
# 为了取 patch 不越界：对三帧一起 padding
pad = patch_size // 2 + search_radius + 2  # 稍微多 pad 一点更稳
Itm1_pad = pad_image(Itm1, pad, border_mode)
It_pad   = pad_image(It,   pad, border_mode)
Itp1_pad = pad_image(Itp1, pad, border_mode)

# frame 列表：便于统一循环
frames = [Itm1_pad, It_pad, Itp1_pad]
frame_names = ["t-1", "t", "t+1"]

# patch 网格（reference patch 在中心帧 It 上取）
# 这里让 reference patch 的左上角坐标落在 [0, H-p] / [0, W-p]
ys = list(range(0, H - patch_size + 1, stride))
xs = list(range(0, W - patch_size + 1, stride))
log_print(f"[GRID] #ys={len(ys)}, #xs={len(xs)}, stride={stride}, patch={patch_size}")
log_print(f"[MATCH] search_radius={search_radius}, top_k={top_k}")

t0 = time.time()
ref_count = 0

# 你可以把所有 3D 组块都存起来，但那会非常大。
# 这里先 debug：只做前 max_patches_debug 个 reference patch，确认逻辑正确。
for y in ys:
    for x in xs:
        # 控制 debug 数量
        if ref_count >= max_patches_debug:
            break

        # 注意：frames 是 padded 图，所以坐标要 +pad
        y_pad = y + pad
        x_pad = x + pad

        # 1) reference patch：来自中心帧 It
        ref_patch = extract_patch(It_pad, y_pad, x_pad, patch_size)

        # 2) 在三帧内、固定窗口搜索：收集全部候选块并算 L2
        cand_list = []  # 每个元素：(dist, frame_id, yy, xx, patch)
        for fid, F in enumerate(frames):
            # 搜索窗口左上角范围（在 padded 坐标上）
            # 我们让候选 patch 的左上角 (yy,xx) 在 [y-R, y+R] 范围内移动
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    yy = y_pad + dy
                    xx = x_pad + dx
                    cand_patch = extract_patch(F, yy, xx, patch_size)
                    dist = l2_dist(ref_patch, cand_patch)
                    # 存“原图坐标”方便 debug 可视化（要减 pad）
                    cand_list.append((dist, fid, yy - pad, xx - pad, cand_patch))

        # 3) 取 Top-K 相似块（全局：跨三帧一起选）
        cand_list.sort(key=lambda t: t[0])
        top = cand_list[:top_k]

        # stack 成 (K, p, p)
        group3d = np.stack([tup[4] for tup in top], axis=0).astype(np.float32)

        # 保存：npy + 拼图
        npy_path = dbg_blocks_dir / f"ref_{ref_count:04d}_y{y}_x{x}_K{top_k}.npy"
        np.save(npy_path, group3d)

        collage = make_collage(group3d, cols=8, scale=8)
        save_gray01_png(dbg_blocks_dir / f"ref_{ref_count:04d}_collage.png", collage)

        # （可选）保存匹配位置示意图：画在中心帧上（也可画在三帧分别）
        if save_match_vis:
            matches_xy = [(tup[2], tup[3]) for tup in top]  # (y, x) in original coords
            vis = draw_search_and_matches(It, (y, x), matches_xy, search_radius, patch_size)
            cv2.imwrite(str(dbg_vis_dir / f"ref_{ref_count:04d}_match_vis.png"), vis)

        # log 一下该 reference 的前几名来源（看是不是合理：一般 t 帧会占多一些）
        src_count = {0: 0, 1: 0, 2: 0}
        for dist, fid, yy, xx, _ in top:
            src_count[fid] += 1
        log_print(
            f"[REF {ref_count:04d}] (y={y}, x={x}) "
            f"TopK src: t-1={src_count[0]}, t={src_count[1]}, t+1={src_count[2]} "
            f"best_dist={top[0][0]:.6e}"
        )

        ref_count += 1

    if ref_count >= max_patches_debug:
        break

dt = time.time() - t0
log_print(f"[DONE] built {ref_count} groups, time={dt:.2f}s, avg={dt/max(ref_count,1):.4f}s/group")
log_print(f"[OUT] dbg_blocks_dir = {dbg_blocks_dir}")
log_print(f"[OUT] dbg_vis_dir    = {dbg_vis_dir}")
log_print("=" * 70)
