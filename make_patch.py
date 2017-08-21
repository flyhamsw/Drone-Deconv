import data
import cv2
import numpy as np
import os
from tqdm import tqdm

ngii_dir = data.get_ngii_dir_all()

patches_dir = 'patches'

patch_size = 224

patch_stride = patch_size

for row in tqdm(ngii_dir):
	name = []
	curr_dataset_name = row[0]
	x_dir = row[1]
	y_dir = row[2]

	x = np.array(cv2.imread(x_dir))
	y = np.array(cv2.imread(y_dir))

	xpath = '%s/%s/x' % (patches_dir, curr_dataset_name)
	ypath = '%s/%s/y' % (patches_dir, curr_dataset_name)

	os.makedirs(xpath)
	os.makedirs(ypath)

	rows = y.shape[0]
	cols = y.shape[1]

	x_data = []
	y_data = []
	y_label = []

	for i in range(0, rows, patch_stride):
		for j in range(0, cols, patch_stride):
			try:
				y_patch = np.array(y[i:i+patch_size, j:j+patch_size])
				if y_patch.shape != (patch_size, patch_size, 3):
					pass
				else:
					y_patch_0 = y_patch

					M = cv2.getRotationMatrix2D((y_patch_0.shape[1]/2, y_patch_0.shape[0]/2), 90, 1)
					y_patch_90 = cv2.warpAffine(y_patch_0, M, (y_patch_0.shape[1], y_patch_0.shape[0]))
					y_patch_180 = cv2.warpAffine(y_patch_90, M, (y_patch_0.shape[1], y_patch_0.shape[0]))
					y_patch_270 = cv2.warpAffine(y_patch_180, M, (y_patch_0.shape[1], y_patch_0.shape[0]))

					yname0 = '%s/NGII_Data_%s_%s_y_0.png' % (ypath, i, j)
					yname90 = '%s/NGII_Data_%s_%s_y_90.png' % (ypath, i, j)
					yname180 = '%s/NGII_Data_%s_%s_y_180.png' % (ypath, i, j)
					yname270 = '%s/NGII_Data_%s_%s_y_270.png' % (ypath, i, j)

					cv2.imwrite(yname0, y_patch_0)
					cv2.imwrite(yname90, y_patch_90)
					cv2.imwrite(yname180, y_patch_180)
					cv2.imwrite(yname270, y_patch_270)

					y_data.append(yname0)
					y_data.append(yname90)
					y_data.append(yname180)
					y_data.append(yname270)

					x_patch = np.array(x[i:i+patch_size, j:j+patch_size])

					x_patch_0 = x_patch

					M = cv2.getRotationMatrix2D((x_patch_0.shape[1]/2, x_patch_0.shape[0]/2), 90, 1)
					x_patch_90 = cv2.warpAffine(x_patch_0, M, (x_patch_0.shape[1], x_patch_0.shape[0]))
					x_patch_180 = cv2.warpAffine(x_patch_90, M, (x_patch_0.shape[1], x_patch_0.shape[0]))
					x_patch_270 = cv2.warpAffine(x_patch_180, M, (x_patch_0.shape[1], x_patch_0.shape[0]))

					xname0 = '%s/NGII_Data_%s_%s_x_0.png' % (xpath, i, j)
					cv2.imwrite(xname0, x_patch_0)
					xname90 = '%s/NGII_Data_%s_%s_x_90.png' % (xpath, i, j)
					cv2.imwrite(xname90, x_patch_90)
					xname180 = '%s/NGII_Data_%s_%s_x_180.png' % (xpath, i, j)
					cv2.imwrite(xname180, x_patch_180)
					xname270 = '%s/NGII_Data_%s_%s_x_270.png' % (xpath, i, j)
					cv2.imwrite(xname270, x_patch_270)

					x_data.append(xname0)
					x_data.append(xname90)
					x_data.append(xname180)
					x_data.append(xname270)

					for p in range(0, 4):
						name.append(curr_dataset_name)
			except Exception as e:
				print(e)
	try:
		data.insert_patch(name, x_data, y_data)
	except Exception as e:
		print(e)
