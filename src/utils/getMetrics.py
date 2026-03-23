import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

img_ori = cv2.imread("data/SIDD_Medium_Srgb/0013_001_S6_03200_01250_3200_L/0013_GT_SRGB_010.PNG")
img2 = cv2.imread("data/SIDD_Medium_Srgb/0013_001_S6_03200_01250_3200_L/0013_GT_SRGB_011.PNG")

# 转 Y 通道
img_ori_y = cv2.cvtColor(img_ori, cv2.COLOR_BGR2YCrCb)[:, :, 0]
img2_y = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:, :, 0]

psnr_val = psnr(img_ori_y, img2_y, data_range=255)
ssim_val = ssim(img_ori_y, img2_y, data_range=255)

print(f"PSNR(Y): {psnr_val:.2f}, SSIM(Y): {ssim_val:.4f}")