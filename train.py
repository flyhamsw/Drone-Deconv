import model
import tensorflow as tf
from pipeline import input_pipeline
import shutil
import os
import data
from tqdm import tqdm

TRAINING_DATASET = 'NGII_training.tfrecords'
VALIDATION_DATASET = 'NGII_validation.tfrecords'
BATCH_SIZE = 8
NUM_EPOCHS = 20

def train(d, batch_size, epoch):
    #Set directory for tensorboard and trained model
    TB_DIR = '/home/lsmjn/Drone-Deconv/tb'
    TRAINED_MODEL_DIR = 'trained_model'
    try:
        shutil.rmtree(TB_DIR)
        shutil.rmtree(TRAINED_MODEL_DIR)
    except Exception as e:
        print(e)
    os.makedirs(TB_DIR)
    os.makedirs(TRAINED_MODEL_DIR)

    #Set saver and merge
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    #Get steps per epoch
    steps = data.get_steps_per_epoch(batch_size, 'training')

    #Start Training
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(TB_DIR + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(TB_DIR + '/test')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
		
        k = 0

        for i in range(0, epoch):
            print('epoch # %d' % i)
            for j in tqdm(range(0, steps)):
                sess.run(d.train_step, feed_dict={d.am_testing: False})
                if k % 100 == 0:
                    summary, _ = sess.run([merged, d.train_step], feed_dict={d.am_testing: False})
                    train_writer.add_summary(summary, k)
                    summary, _ = sess.run([d.xe_valid_summary, d.cross_entropy_valid], feed_dict={d.am_testing: True})
                    test_writer.add_summary(summary, k)
                k = k + 1
            
        coord.request_stop()
        coord.join(threads)
        save_path = saver.save(sess, "/home/lsmjn/Drone-Deconv/trained_model/Drone_CNN.ckpt")
        print('Model saved in file: %s' % save_path)
        train_writer.close()

if __name__ == '__main__':
    x_batch_train, y_batch_train = input_pipeline(TRAINING_DATASET, BATCH_SIZE, NUM_EPOCHS)
    x_batch_validation, y_batch_validation = input_pipeline(VALIDATION_DATASET, BATCH_SIZE, NUM_EPOCHS)
    d = model.Deconv(x_batch_train, y_batch_train, x_batch_validation, y_batch_validation)
    train(d, BATCH_SIZE, NUM_EPOCHS)