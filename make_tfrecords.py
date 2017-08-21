'''
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
'''
import cv2
import numpy as np
import tensorflow as tf
import data
from tqdm import tqdm

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def build_tfrecords(target_dataset):
    conn, cur = data.get_db_connection()
    filename_pairs, _, _ = data.get_patch_all(conn, cur, target_dataset)

    tfrecords_filename = 'NGII_%s.tfrecords' % target_dataset

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for img_path, annotation_path in tqdm(filename_pairs):

        img = np.array(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        annotation = np.array(cv2.cvtColor(cv2.imread(annotation_path), cv2.COLOR_BGR2RGB))

        img_raw = img.tostring()
        annotation_raw = annotation.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw)}))

        writer.write(example.SerializeToString())

    writer.close()

if __name__ == '__main__':
    build_tfrecords('training')
    build_tfrecords('validation')
    build_tfrecords('test')
