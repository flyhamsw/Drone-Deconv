import model
import tensorflow as tf
import cv2

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
            cv2.imwrite('Ganghwa_p.jpg', result[batch_idx]*255)
            
    print('Image Segmentation Complete!')

if __name__ == '__main__':
    print('Preparing Image...')
    #drone_image = cv2.cvtColor(cv2.imread('drone_dataset/Ganghwa/DSC00076.JPG'), cv2.COLOR_BGR2RGB)
    drone_image = cv2.imread('drone_dataset/Ganghwa/DSC00076.JPG')
    drone_image = cv2.resize(drone_image, None, fx=0.0625, fy=0.0625, interpolation=cv2.INTER_AREA)
    print(drone_image.shape)
    drone_image = drone_image[0:223, 0:223]
    
    cv2.imwrite('Ganghwa.jpg', drone_image)
    
    x_drone = tf.placeholder('float', shape=[None, drone_image.shape[0], drone_image.shape[1], 3])
    d = model.Deconv(x_drone, x_drone, x_drone, x_drone)
    
    predict(d, drone_image)
