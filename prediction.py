import model
import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm
import sys

PATCH_SIZE = 224 * 8

def prepare_patches(drone_image_dir):
    patches = []
    drone_image = cv2.imread(drone_image_dir)

    width = drone_image.shape[0]
    height = drone_image.shape[1]

    width_iter = (drone_image.shape[0] // PATCH_SIZE + 1)
    height_iter = (drone_image.shape[1] // PATCH_SIZE + 1)

    for i in range(0, width_iter):
        for j in range(0, height_iter):
            patch = drone_image[PATCH_SIZE*i:PATCH_SIZE*(i+1), PATCH_SIZE*j:PATCH_SIZE*(j+1), :]
            if patch.shape != (PATCH_SIZE, PATCH_SIZE, 3):
                im_concat = np.concatenate((patch, np.zeros((PATCH_SIZE - patch.shape[0], patch.shape[1], 3))))
                im_concat = np.concatenate((im_concat, np.zeros((PATCH_SIZE, PATCH_SIZE - patch.shape[1], 3))), axis = 1)
                patch = im_concat
            patches.append(patch)
    return patches, width_iter, height_iter, width, height

def predict(d, patches, width_iter, height_iter, width, height):
    otherwise_dir_list = []
    building_dir_list = []
    road_dir_list = []

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, 'trained_model/Drone_Deconv.ckpt')

        k = 0

        for patch in tqdm(patches):
            result = sess.run(d.y_soft, feed_dict={d.am_testing: False, d.x_batch_train: [patch]})

            for batch_idx in range(0, len(result)):
                otherwise_dir = 'segmentation_result/%d_otherwise.png' % k
                building_dir = 'segmentation_result/%d_building.png' % k
                road_dir = 'segmentation_result/%d_road.png' % k

                otherwise_dir_list.append(otherwise_dir)
                building_dir_list.append(building_dir)
                road_dir_list.append(road_dir)

                cv2.imwrite(otherwise_dir, result[batch_idx][:, :, 0] * 255)
                cv2.imwrite(building_dir, result[batch_idx][:, :, 2] * 255)
                cv2.imwrite(road_dir, result[batch_idx][:, :, 1] * 255)

            k = k + 1

    result_list = []

    for dir_list in [otherwise_dir_list, building_dir_list, road_dir_list]:
        k = 0
        result = np.zeros((PATCH_SIZE * width_iter, PATCH_SIZE * height_iter, 3))
        for i in range(0, width_iter):
            for j in range(0, height_iter):
                result[PATCH_SIZE*i:PATCH_SIZE*(i+1), PATCH_SIZE*j:PATCH_SIZE*(j+1)] = cv2.imread(dir_list[k])
                print(dir_list[k])
                k = k + 1
        result_list.append(result[0:width, 0:height, :])

    cv2.imwrite('segmentation_result_otherwise.png', result_list[0])
    cv2.imwrite('segmentation_result_building.png', result_list[1])
    cv2.imwrite('segmentation_result_road.png', result_list[2])

if __name__ == '__main__':
    filename = sys.argv[1]
    patches, width_iter, height_iter, width, height = prepare_patches(filename)

    x_drone = tf.placeholder(tf.float32, shape=[None, PATCH_SIZE, PATCH_SIZE, 3])
    d = model.Deconv(x_drone, prediction=True, num_of_class=3)
    
    predict(d, patches, width_iter, height_iter, width, height)