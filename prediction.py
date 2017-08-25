import model
import tensorflow as tf
from pipeline import input_pipeline
import shutil
import os
import data

TEST_DATASET = 'NGII_test.tfrecords'
BATCH_SIZE = 8
NUM_EPOCHS = 1

def predict(d, batch_size, epoch):
    #Get steps per epoch
    steps = data.get_steps_per_epoch(batch_size, 'test)

    #Start Training
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
		
        k = 0

        for i in range(0, epoch):
            for j in range(0, steps):
                result = sess.run(d.y_soft, feed_dict={d.am_testing: False})
                k = k + 1
            
        coord.request_stop()
        coord.join(threads)
        save_path = saver.save(sess, "/home/lsmjn/Drone-Deconv/trained_model/Drone_CNN.ckpt")
        print('Model saved in file: %s' % save_path)
        train_writer.close()

if __name__ == '__main__':
    x_batch_test, y_batch_test = input_pipeline(TEST_DATASET, BATCH_SIZE, NUM_EPOCHS)
    d = model.Deconv(x_batch_train, y_batch_train, None, None)
    predict(d, BATCH_SIZE, NUM_EPOCHS)