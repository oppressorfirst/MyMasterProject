import cv2
import numpy as np
import os

# ============================================================
# CRVD 数据集相机参数（Sony IMX385，室内场景）
# ============================================================

# 黑白电平（来自官方 README）
BLACK_LEVEL = 240.0
WHITE_LEVEL = 4095.0  # 2**12 - 1

# Sony 相机白平衡增益（R, G, B）
# 来源：CRVD/RViDeNet 官方代码中 SID Sony 数据集的典型值
# 如果你有 EXIF 可以替换为实际值
WB_GAINS = np.array([2.394, 1.0, 1.597], dtype=np.float32)  # R, G, B

# 色彩矩阵 CCM：Camera RGB → linear sRGB
# 来源：Sony IMX385 / A7S 系列典型 DNG ColorMatrix2（D65 光源）
# 格式：每行是 [sR, sG, sB] 对应一个输出通道的权重
CCM = np.array([
    [ 1.9712, -0.7503, -0.2209],
    [-0.2491,  1.4727, -0.2236],
    [ 0.0172, -0.4864,  1.4692]
], dtype=np.float32)


def apply_srgb_gamma(linear: np.ndarray) -> np.ndarray:
    """严格 sRGB gamma 曲线（IEC 61966-2-1），比简单 1/2.2 更准确"""
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055
    )


def raw_bayer_to_srgb(raw_img: np.ndarray) -> np.ndarray:
    """
    将单通道 RAW Bayer TIFF（uint16）转换为 sRGB uint8。

    流程：
      1. 黑白电平归一化
      2. 去马赛克（GBRG）
      3. 白平衡
      4. 色彩矩阵（Camera RGB → linear sRGB）
      5. sRGB Gamma 校正
      6. 转为 uint8
    """
    # --- Step 1: 黑白电平归一化 ---
    raw_f = raw_img.astype(np.float32)
    raw_f = (raw_f - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL)
    raw_f = np.clip(raw_f, 0.0, 1.0)

    # --- Step 2: 去马赛克 ---
    # CRVD Bayer Pattern = GBRG
    # OpenCV 的命名约定：COLOR_BAYER_XX2YY 中 XX 是左上角 2x2 的前两个像素
    # GBRG 左上角是 G,B → 对应 OpenCV 的 COLOR_BAYER_GB2BGR（注意不是 GR）
    raw_u16 = (raw_f * 65535).astype(np.uint16)
    bgr_u16 = cv2.cvtColor(raw_u16, cv2.COLOR_BAYER_GR2BGR)
    bgr_f = bgr_u16.astype(np.float32) / 65535.0  # shape: H×W×3, 顺序 B,G,R

    # --- Step 3: 白平衡 ---
    # bgr_f 通道顺序是 B, G, R
    bgr_f[:, :, 2] *= WB_GAINS[0]  # R
    bgr_f[:, :, 1] *= WB_GAINS[1]  # G（通常为 1.0）
    bgr_f[:, :, 0] *= WB_GAINS[2]  # B
    bgr_f = np.clip(bgr_f, 0.0, 1.0)

    # --- Step 4: 色彩矩阵（CCM）Camera RGB → linear sRGB ---
    # 转为 RGB 顺序做矩阵乘法，再转回 BGR
    rgb_f = bgr_f[:, :, ::-1]  # BGR → RGB
    h, w, _ = rgb_f.shape
    rgb_flat = rgb_f.reshape(-1, 3)          # (N, 3)
    rgb_flat = (CCM @ rgb_flat.T).T          # (N, 3)
    rgb_f = np.clip(rgb_flat.reshape(h, w, 3), 0.0, 1.0)

    # --- Step 5: sRGB Gamma ---
    srgb_f = apply_srgb_gamma(rgb_f)

    # --- Step 6: BGR uint8 输出（OpenCV 写图用 BGR）---
    bgr_out = (srgb_f[:, :, ::-1] * 255.0).astype(np.uint8)
    return bgr_out


def process_crvd_recursive_forward_isp(input_dir, output_dir, is_denoised=False):
    """
    递归遍历 input_dir 下所有 .tif/.tiff 文件，执行正向 ISP 转换为 sRGB，
    保持原始目录结构保存到 output_dir。

    参数：
        input_dir   : 输入目录（原始 RAW 或降噪后的单通道 RAW）
        output_dir  : 输出目录
        is_denoised : 是否是降噪后的 RAW 图像。
                      True  → 假设输入仍为单通道 RAW，执行 ISP（与原始 RAW 完全相同的流程）
                      False → 默认行为，相同处理

    注意：
        - 如果你的降噪网络输出的已经是 sRGB 多通道图像，请直接跳过本脚本，
          不需要再做任何 ISP。
        - 如果你的降噪网络在 RAW 域降噪后输出仍是单通道 Bayer RAW，
          则可以用 is_denoised=True 调用本函数做 ISP，效果与原始 RAW 一致。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if not filename.lower().endswith(('.tif', '.tiff')):
                continue

            input_path = os.path.join(root, filename)
            rel_path = os.path.relpath(root, input_dir)
            target_subdir = output_dir if rel_path == '.' else os.path.join(output_dir, rel_path)
            os.makedirs(target_subdir, exist_ok=True)

            output_name = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(target_subdir, output_name)

            raw_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if raw_img is None:
                print(f"❌ 读取失败: {input_path}")
                continue

            tag = "[降噪后RAW]" if is_denoised else "[原始RAW]"

            if len(raw_img.shape) == 2:
                # 单通道 → 执行完整 ISP 流程（原始 RAW 和降噪后 RAW 都走这里）
                try:
                    srgb = raw_bayer_to_srgb(raw_img)
                    cv2.imwrite(output_path, srgb)
                    print(f"✅ {tag} ISP 完成: {os.path.join(rel_path, filename)} → {output_name}")
                except Exception as e:
                    print(f"❌ 转换失败 {input_path}: {e}")

            elif len(raw_img.shape) == 3:
                # 多通道图像：说明已经是 RGB/BGR 图（例如降噪网络直接输出 sRGB）
                # 本脚本不做任何处理，直接转存，避免二次 ISP 破坏图像
                print(f"⏩ {tag} 已是多通道图像，跳过 ISP 直接转存: {os.path.join(rel_path, filename)}")
                # 如果原始已是 uint8 或 uint16，直接保存
                if raw_img.dtype == np.uint16:
                    raw_img_8bit = (raw_img / 256).astype(np.uint8)
                    cv2.imwrite(output_path, raw_img_8bit)
                else:
                    cv2.imwrite(output_path, raw_img)
            else:
                print(f"⚠️ 未知图像格式，跳过: {input_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRVD RAW TIFF → sRGB PNG 转换工具")
    parser.add_argument("--input",      default="./out/results/CRVD",       help="输入目录（原始 RAW 或降噪后 RAW）")
    parser.add_argument("--output",     default="./out/results/CRVD_srgb",  help="输出目录")
    parser.add_argument("--denoised",   action="store_true",          help="输入是降噪后的 RAW（加此参数仅影响日志标记，ISP 流程相同）")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"输入目录 : {args.input}")
    print(f"输出目录 : {args.output}")
    print(f"模式     : {'降噪后RAW → sRGB' if args.denoised else '原始RAW → sRGB'}")
    print(f"{'='*60}")

    process_crvd_recursive_forward_isp(args.input, args.output, is_denoised=args.denoised)
    print("\n✅ 全部处理完成！")