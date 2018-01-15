'''
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
'''
import cv2
import tensorflow as tf
import data
from tqdm import tqdm

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

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def build_tfrecords(target_dataset, augment_op=False):
    conn, cur = data.get_db_connection()
    filename_pairs, _, _ = data.get_patch_all(conn, cur, target_dataset)
    tfrecords_filename = '/media/lsmjn/56fcc20e-a0ee-45e0-8df1-bf8b2e9a43b2/tfrecords/NGII_%s.tfrecords' % target_dataset
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for img_path, annotation_path in tqdm(filename_pairs):
        img = cv2.imread(img_path)
        annotation = cv2.imread(annotation_path)

        if augment_op:
            augmented_img_list = augment_image(img, sess)
            augmented_annotation_list = augment_image(annotation, sess)

            for i in range(0, 7):
                img_raw = augmented_img_list[i].tostring()
                annotation_raw = augmented_annotation_list[i][:,:,0].tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': _bytes_feature(img_raw),
                    'mask_raw': _bytes_feature(annotation_raw)}))

                writer.write(example.SerializeToString())
        else:
            img_raw = img.tostring()
            annotation_raw = annotation[:,:,0].tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(img_raw),
                'mask_raw': _bytes_feature(annotation_raw)}))

            writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    build_tfrecords('training')
    build_tfrecords('validation')
    build_tfrecords('test')

