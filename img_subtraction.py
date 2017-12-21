import cv2
import numpy as np

def img_subtraction(p_dir, y_dir):
    p = np.array(cv2.imread(p_dir), dtype=float)[:,:,2]
    y = np.array(cv2.imread(y_dir), dtype=float)[:,:,2]

    subtraction = p - y

    return subtraction