import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
import data, model

PATCH_SIZE = 224

def create_prediction_result_patches(patch_filename_list, num_of_class=3):
    x_drone = tf.placeholder(tf.float32, shape=[None, PATCH_SIZE, PATCH_SIZE, 3])
    d = model.Deconv(x_drone, prediction=True, num_of_class=num_of_class)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, 'trained_model/Drone_Deconv.ckpt')

        i = 0
        k = 0
        x_filename_list = []
        y_index_filename_list = []
        p_filename_list = []
        batch = []

        for patch_dir_pair in tqdm(patch_filename_list):
            x_filename = 'prediction_result/%d_x.png' % i
            y_index_filename = 'prediction_result/%d_y_index.png' % i
            i = i + 1
            cv2.imwrite(x_filename, cv2.imread(patch_dir_pair[0]))
            cv2.imwrite(y_index_filename, cv2.imread(patch_dir_pair[1]))

            x_filename_list.append(x_filename)
            y_index_filename_list.append(y_index_filename)

            batch.append(cv2.cvtColor(cv2.imread(patch_dir_pair[0]), cv2.COLOR_BGR2RGB))

            if len(batch) == 8 or (len(patch_filename_list) - k < 8):
                result_batch = sess.run(d.y_soft, feed_dict={d.am_testing: False, d.x_batch_train: batch})
                for result in result_batch:
                    p_filename_all_categories = []
                    for current_category in range(0, num_of_class):
                        p_filename_all_categories.append('prediction_result/%d_p_%d.png' % (k, current_category))
                        cv2.imwrite(p_filename_all_categories, result[:,:,current_category])
                    p_filename_list.append(p_filename_all_categories)
                    k = k + 1
                    batch = []

    return x_filename_list, y_index_filename_list, p_filename_list

def y_index_one_hot(y_index):
    result = np.zeros(y_index.shape)
    y_index = y_index[:,:,0]

    for i in range(0, y_index.shape[0]):
        for j in range(0, y_index.shape[1]):
            result[i, j, y_index[i, j]] = 1

    return result == 1

def recall_precision(y_index_filename_list, p_filename_list, interest_category):
    with open('recall-precision_ch_%d.csv' % interest_category, 'w') as f:
        for t in np.arange(0.05, 1, 0.05):
            n_TP_sum = 0
            n_FP_sum = 0
            n_FN_sum = 0
            
            for i in tqdm(range(0, len(p_filename_list))):
                ground_truth = cv2.imread(y_index_filename_list[i])
                ground_truth = y_index_one_hot(ground_truth)
                prediction = cv2.imread(p_filename_list[i][interest_category])[:, :, interest_category]
                prediction = prediction > t
                
                n_TP_sum = n_TP_sum + np.sum((ground_truth == True) & (prediction == True))
                n_FP_sum = n_FP_sum + np.sum((ground_truth == False) & (prediction == True))
                n_FN_sum = n_FN_sum + np.sum((ground_truth == True) & (prediction == False))
                
            recall = n_TP_sum / (n_TP_sum + n_FN_sum)
            precision = n_TP_sum / (n_TP_sum + n_FP_sum)
            
            line = '%f,%f,%f\n' % (t, recall, precision)
            print(line)
            f.write(line)

if __name__=='__main__':
    conn, cur = data.get_db_connection()
    patch_filename_list, _, _ = data.get_patch_all(conn, cur, 'test')

    x_filename_list, y_index_filename_list, p_filename_list = create_prediction_result_patches(patch_filename_list)

    for i in range(0, 3):
        recall_precision(y_index_filename_list, p_filename_list, i)