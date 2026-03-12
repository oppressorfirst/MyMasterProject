import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2

# 添加 utils 文件夹到 Python 路径
from utils import getMetrics, AI_Metrics
import time


original_path = "/Users/oppressor/Desktop/MyMasterProject/data/classic_photo/lena_gray.png"
original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
original = original.astype(np.float32)
denoised_img_path = "/Users/oppressor/Desktop/MyMasterProject/out/images/Copy of 03_denoised.png"
denoised_img = cv2.imread(denoised_img_path, cv2.IMREAD_GRAYSCALE)
denoised_img = denoised_img.astype(np.float32)


denoised_metrics = getMetrics.calculate_metrics(original.astype(np.uint8), denoised_img)
print(f"去噪后PSNR: {denoised_metrics['PSNR']:.2f} dB")
print(f"去噪后SSIM: {denoised_metrics['SSIM']:.4f}")
print(f"去噪后MSE: {denoised_metrics['MSE']:.2f}")