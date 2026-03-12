import cv2

def read_png_get_y(path):
    """
    读取PNG并返回Y通道
    """
    img = cv2.imread(path)  # BGR
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycbcr[:, :, 0]
    return y
