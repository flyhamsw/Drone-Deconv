import cv2
import numpy as np

kernels = []

#interest_ch: 0-building
def img_subtraction(p_dir, y_dir, interest_ch=0, threshold=0.75):
    p = cv2.imread(p_dir)[:,:,interest_ch] > threshold * 255
    p = np.array(p, dtype=float) * 255
    y = np.array(cv2.imread(y_dir), dtype=float)[:,:,interest_ch] * 255

    subtraction = p - y

    cv2.imwrite('subtraction/p.png', p)
    cv2.imwrite('subtraction/subtraction.png', subtraction)
    cv2.imwrite('subtraction/y.png', y)

    i = 0
    for kernel in kernels:
        opening = cv2.dilate(cv2.erode(subtraction, kernel, iterations=1), kernel, iterations=1)
        cv2.imwrite('subtraction/%d.png' % i, opening)
        i = i + 1

if __name__ == '__main__':
    for i in range(3, 20, 2):
        kernels.append(np.ones((i, i), np.uint8))
    img_subtraction('segmentation_result_building.png', 'gangnam_y.png')