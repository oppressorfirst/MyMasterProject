
import cv2
import os

def convert_raw_to_720p(input_path, output_path):
    # 1. 读取 16-bit 的 TIFF RAW 图 (必须加 IMREAD_UNCHANGED)
    raw_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if raw_img is None:
        print(f"读取失败，请检查路径: {input_path}")
        return

    H, W = raw_img.shape
    print(f"原图尺寸: {W}x{H}")

    target_H, target_W = 720, 1280

    # 2. 计算中心裁剪的起始坐标
    # 1080 - 720 = 360 -> 起点 180
    # 1920 - 1280 = 640 -> 起点 320
    y_start = (H - target_H) // 2
    x_start = (W - target_W) // 2

    # 3. 硬件级断言：确保起点是偶数，完美保留 GBRG Bayer 阵列相位！
    assert y_start % 2 == 0 and x_start % 2 == 0, "裁剪起点必须为偶数以保留 Bayer 相位！"

    y_end = y_start + target_H
    x_end = x_start + target_W

    # 4. 执行裁剪
    raw_720p = raw_img[y_start:y_end, x_start:x_end]

    # 5. 保存为新的 16-bit TIFF
    cv2.imwrite(output_path, raw_720p)
    print(f"成功保存 720p 图像至: {output_path} (尺寸: {raw_720p.shape[1]}x{raw_720p.shape[0]})")


if __name__ == "__main__":
    # 输入路径
    NOISY = 'data/CRVD/noisy/scene1/ISO3200/frame1_noisy0.tiff'
    CLEAN = 'data/CRVD/noisy/scene1/ISO3200/frame1_clean.tiff'
    
    # 输出路径 (建议加个后缀区分)
    NOISY_720P = 'data/CRVD/noisy/scene1/ISO3200/frame1_noisy0_720p.tiff'
    CLEAN_720P = 'data/CRVD/noisy/scene1/ISO3200/frame1_clean_720p.tiff'

    convert_raw_to_720p(NOISY, NOISY_720P)
    convert_raw_to_720p(CLEAN, CLEAN_720P)