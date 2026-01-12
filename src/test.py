import numpy as np
import matplotlib.pyplot as plt
import cv2

original_path = "../data/classic_photo/lena_gray.png"
noisy_path = "../data/classic_photo_AWGN_sigma20_seed123456/lena_gray_sigma20_seed123456.png"

original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
patch_r = 90
padded_img = np.pad(noisy_img, pad_width=patch_r, mode='reflect')
i_row = 64
i_col = 64


i_row_pad = i_row + patch_r
i_col_pad = i_col + patch_r

patch_i = padded_img[0: 540,0 :240]
plt.figure(figsize=(8, 4))
plt.imshow(patch_i, cmap='gray')
plt.show()