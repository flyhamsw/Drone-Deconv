import model
import tensorflow as tf
import cv2
import numpy as np

def predict(d, drone_image):
    saver = tf.train.Saver()

    #Start Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('Restoring Model...')
        saver.restore(sess, 'trained_model/Drone_CNN.ckpt')

        print('Image Segmentation...')
        result = sess.run(d.y_soft, feed_dict={d.am_testing: False, d.x_batch_train: [drone_image]})
        
        for batch_idx in range(0, len(result)):
            print('Saving Result...')
            cv2.imwrite('ganghwa_crop_p.jpg', result[batch_idx]*255)
            
    print('Image Segmentation Complete!')

if __name__ == '__main__':
    print('Preparing Image...')
    '''
    drone_image = np.asarray(cv2.imread('drone_dataset/x_sinjeong.png'))
    print(drone_image.shape)
    drone_image = drone_image[0:224*9, 0:224*9]
    print(drone_image.shape)
    
    cv2.imwrite('Sinjeong.png', drone_image)
    '''
    
    drone_image = cv2.imread('ganghwa_crop.png')
    
    x_drone = tf.placeholder('float', shape=[None, drone_image.shape[0], drone_image.shape[1], 3])
    d = model.Deconv(x_drone, x_drone, x_drone, x_drone)
    
    predict(d, drone_image)
