import model
import tensorflow as tf
from pipeline import input_pipeline
import shutil
import os
import data
import cv2
from tqdm import tqdm

TEST_DATASET = 'NGII_test.tfrecords'
BATCH_SIZE = 8
NUM_EPOCHS = 1

def predict(d, batch_size, epoch):
    #Get steps per epoch
    steps = data.get_steps_per_epoch(batch_size, 'test')

    saver = tf.train.Saver()

    #Start Training
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        k = 0

        saver.restore(sess, 'trained_model/Drone_CNN.ckpt')

        for i in range(0, epoch):
            for j in tqdm(range(0, steps)):
                x_image, ground_truth, result = sess.run([d.x_image, d.y_, d.y_soft], feed_dict={d.am_testing: False})
                for batch_idx in range(0, len(x_image)):
                    cv2.imwrite('prediction_result/%d_x.png' % k, x_image[batch_idx])
                    cv2.imwrite('prediction_result/%d_y.png' % k, ground_truth[batch_idx]*255)
                    cv2.imwrite('prediction_result/%d_p.png' % k, result[batch_idx]*255)
                    k = k + 1

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    x_batch_test, y_batch_test = input_pipeline(TEST_DATASET, BATCH_SIZE, NUM_EPOCHS)
    d = model.Deconv(x_batch_test, y_batch_test, x_batch_test, y_batch_test)
    predict(d, BATCH_SIZE, NUM_EPOCHS)
