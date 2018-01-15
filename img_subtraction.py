import cv2
import numpy as np

kernels = []

kernels.append(np.ones((3, 3), np.uint8))
kernels.append(np.ones((5, 5), np.uint8))
kernels.append(np.ones((7, 7), np.uint8))

def img_subtraction(p_dir, y_dir, interest_ch=2):
    p = np.array(cv2.imread(p_dir), dtype=float)[:,:,interest_ch] / 255
    y = np.array(cv2.imread(y_dir), dtype=float)[:,:,interest_ch]

    subtraction = p - y

    cv2.imwrite('p.png', p * 225)
    cv2.imwrite('y.png', y)

    i = 0

    print(p.shape)
    print(y.shape)

    for kernel in kernels:
        opening = cv2.dilate(cv2.erode(subtraction, kernel, iterations=1), kernel, iterations=1) * 255
        cv2.imwrite(str(i) + '.png', opening)
        i = i + 1

    return subtraction

if __name__ == '__main__':
    img_subtraction('segmentation_result_building.png', 'yeosu_y.png')