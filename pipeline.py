import tensorflow as tf

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string),
        }
    )

    # must be read back as uint8 here
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    segmentation = tf.decode_raw(features['mask_raw'], tf.uint8)

    image = tf.reshape(image,[224,224,3])
    segmentation = tf.reshape(segmentation,[224,224])

    rgb = tf.cast(image, tf.float32)
    mask = tf.cast(segmentation, tf.int32)

    return rgb, mask

def input_pipeline(filenames, batch_size, num_epochs):
    filename_queue = tf.train.string_input_producer([filenames])

    image, label = read_and_decode(filename_queue)

    num_threads = 2

    min_after_dequeue = 1000
    capacity = min_after_dequeue + (num_threads + 10) * batch_size

    images_batch, labels_batch = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size,
        enqueue_many=False, shapes=None,
        allow_smaller_final_batch=True,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue, num_threads=num_threads)
    return images_batch, labels_batch
