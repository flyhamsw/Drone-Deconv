import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
import data, model

PATCH_SIZE = 224

def create_prediction_result_patches(x_patch_filename_list):
    x_drone = tf.placeholder(tf.float32, shape=[None, PATCH_SIZE, PATCH_SIZE, 3])
    d = model.Deconv(x_drone, prediction=True, num_of_class=3)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, 'trained_model/Drone_Deconv.ckpt')

        k = 0
        prediction_result_list = []
        batch = []

        for patch_name in tqdm(x_patch_filename_list):
            batch.append(cv2.imread(patch_name))
            if len(batch) == 8 or (len(x_patch_filename_list) - k < 8):
                result_batch = sess.run(d.y_soft, feed_dict={d.am_testing: False, d.x_batch_train: batch})
                for result in result_batch:
                    result_filename = 'prediction_result/%d_p.png' % k
                    prediction_result_list.append(result_filename)
                    cv2.imwrite(result_filename, result * 255)
                    k = k + 1
                    batch = []

    return prediction_result_list

def recall_precision(y_patch_filename_list, prediction_result_list, interest_ch):
    with open('recall-precision_ch_%d.csv' % interest_ch, 'w') as f:
        for t in np.arange(0.05, 1, 0.05):
            n_TP_sum = 0
            n_FP_sum = 0
            n_FN_sum = 0
            
            for i in tqdm(range(0, len(y_patch_filename_list))):
                ground_truth = cv2.imread(y_patch_filename_list[i])[:, :, interest_ch]
                ground_truth = ground_truth == 1
                prediction = cv2.imread(prediction_result_list[i])[:,:,interest_ch] / 255
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
    _, x_patch_filename_list, y_patch_filename_list = data.get_patch_all(conn, cur, 'test')

    prediction_result_list = create_prediction_result_patches(x_patch_filename_list)

    for i in range(0, 3):
        recall_precision(y_patch_filename_list, prediction_result_list, i)