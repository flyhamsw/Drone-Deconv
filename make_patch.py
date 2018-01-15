import data
import cv2
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf

ngii_dir = data.get_ngii_dir_all()
patches_dir = 'patches'
patch_size = 224
patch_stride = patch_size

img = tf.placeholder(tf.float32, shape=(224, 224, 3))
img_flip_LR = tf.image.flip_left_right(img)
img_flip_UD = tf.image.flip_up_down(img)
img_flip_TR = tf.image.transpose_image(img)
img_rot_90 = tf.image.rot90(img, 1)
img_rot_180 = tf.image.rot90(img, 2)
img_rot_270 = tf.image.rot90(img, 3)

def augment_image(original_img, sess):
    augmented_images = sess.run([img, img_flip_LR, img_flip_UD, img_flip_TR, img_rot_90, img_rot_180, img_rot_270], feed_dict={img: original_img})
    return augmented_images

def make_patch(sess, floor=False, num_floor=10):
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

        for i in range(0, rows, patch_stride):
            for j in range(0, cols, patch_stride):
                y_patch = np.array(y[i:i+patch_size, j:j+patch_size])

                if floor:
                    y_patch = (y_patch <= num_floor) & (y_patch != 0)

                if y_patch.shape != (patch_size, patch_size, 3):
                    pass
                else:
                    augmented_y = augment_image(y_patch, sess)
                    for num_aug in range(0, 7):
                        yname = '%s/NGII_Data_%s_%s_y_%d.png' % (ypath, i, j, num_aug)
                        cv2.imwrite(yname, augmented_y[num_aug])
                        y_data.append(yname)

                    x_patch = np.array(x[i:i+patch_size, j:j+patch_size])
                    augmented_x = augment_image(x_patch, sess)
                    for num_aug in range(0, 7):
                        xname = '%s/NGII_Data_%s_%s_x_%d.png' % (xpath, i, j, num_aug)
                        cv2.imwrite(xname, augmented_x[num_aug])
                        x_data.append(xname)
                        name.append(curr_dataset_name)

        data.insert_patch(name, x_data, y_data)

if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    make_patch(sess)


