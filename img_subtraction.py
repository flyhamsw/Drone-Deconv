import cv2
import numpy as np

p = np.array(cv2.imread('p.png'), dtype=float)
y = np.array(cv2.imread('y.png'), dtype=float)

subtraction = p - y

cv2.imwrite('subtraction.png', subtraction)