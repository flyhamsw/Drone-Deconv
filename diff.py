import cv2
import numpy as np

def diff(img_p, img_y):
    prediction = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
    gt = cv2.cvtColor(cv2.imread(img_y), cv2.COLOR_BGR2RGB)
	diff = np.maximum(np.array(pred, dtype='float') - np.array(gt, dtype='float'), np.zeros(diff.shape))