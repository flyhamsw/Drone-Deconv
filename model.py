import tensorflow as tf
import CNN as c

class Deconv:
    def __init__(self, x_batch_train, y_batch_train, x_batch_validation, y_batch_validation, lr=0.0001):
        self.lr = lr

        self.am_testing = tf.placeholder(dtype=bool,shape=())

        self.x_batch_train = x_batch_train
        self.y_batch_train = y_batch_train
        self.x_batch_validation = x_batch_validation
        self.y_batch_validation = y_batch_validation

        self.x_image = tf.cond(self.am_testing, lambda: self.x_batch_validation, lambda: self.x_batch_train)
        self.y_ = tf.cond(self.am_testing, lambda: self.y_batch_validation, lambda: self.y_batch_train)

        self.expected = tf.expand_dims(self.y_, -1)

        self.conv_1_1 = c.conv_layer(self.x_image, [3, 3, 3, 64], 64, 'conv_1_1')
        self.conv_1_2 = c.conv_layer(self.conv_1_1, [3, 3, 64, 64], 64, 'conv_1_2')

        self.pool_1, self.pool_1_argmax = c.pool_layer(self.conv_1_2)

        self.conv_2_1 = c.conv_layer(self.pool_1, [3, 3, 64, 128], 128, 'conv_2_1')
        self.conv_2_2 = c.conv_layer(self.conv_2_1, [3, 3, 128, 128], 128, 'conv_2_2')

        self.pool_2, self.pool_2_argmax = c.pool_layer(self.conv_2_2)

        self.conv_3_1 = c.conv_layer(self.pool_2, [3, 3, 128, 256], 256, 'conv_3_1')
        self.conv_3_2 = c.conv_layer(self.conv_3_1, [3, 3, 256, 256], 256, 'conv_3_2')
        self.conv_3_3 = c.conv_layer(self.conv_3_2, [3, 3, 256, 256], 256, 'conv_3_3')

        self.pool_3, self.pool_3_argmax = c.pool_layer(self.conv_3_3)

        self.conv_4_1 = c.conv_layer(self.pool_3, [3, 3, 256, 512], 512, 'conv_4_1')
        self.conv_4_2 = c.conv_layer(self.conv_4_1, [3, 3, 512, 512], 512, 'conv_4_2')
        self.conv_4_3 = c.conv_layer(self.conv_4_2, [3, 3, 512, 512], 512, 'conv_4_3')

        self.pool_4, self.pool_4_argmax = c.pool_layer(self.conv_4_3)

        self.conv_5_1 = c.conv_layer(self.pool_4, [3, 3, 512, 512], 512, 'conv_5_1')
        self.conv_5_2 = c.conv_layer(self.conv_5_1, [3, 3, 512, 512], 512, 'conv_5_2')
        self.conv_5_3 = c.conv_layer(self.conv_5_2, [3, 3, 512, 512], 512, 'conv_5_3')

        self.pool_5, self.pool_5_argmax = c.pool_layer(self.conv_5_3)

        self.fc_6 = c.conv_layer(self.pool_5, [7, 7, 512, 4096], 4096, 'fc_6')
        self.fc_7 = c.conv_layer(self.fc_6, [1, 1, 4096, 4096], 4096, 'fc_7')

        self.deconv_fc_6 = c.deconv_layer(self.fc_7, [7, 7, 512, 4096], 512, 'fc6_deconv')

        self.unpool_5 = c.unpool(self.deconv_fc_6, self.pool_5_argmax)

        self.deconv_5_3 = c.deconv_layer(self.unpool_5, [3, 3, 512, 512], 512, 'deconv_5_3')
        self.deconv_5_2 = c.deconv_layer(self.deconv_5_3, [3, 3, 512, 512], 512, 'deconv_5_2')
        self.deconv_5_1 = c.deconv_layer(self.deconv_5_2, [3, 3, 512, 512], 512, 'deconv_5_1')

        self.unpool_4 = c.unpool(self.deconv_5_1, self.pool_4_argmax)

        self.deconv_4_3 = c.deconv_layer(self.unpool_4, [3, 3, 512, 512], 512, 'deconv_4_3')
        self.deconv_4_2 = c.deconv_layer(self.deconv_4_3, [3, 3, 512, 512], 512, 'deconv_4_2')
        self.deconv_4_1 = c.deconv_layer(self.deconv_4_2, [3, 3, 256, 512], 256, 'deconv_4_1')

        self.unpool_3 = c.unpool(self.deconv_4_1, self.pool_3_argmax)

        self.deconv_3_3 = c.deconv_layer(self.unpool_3, [3, 3, 256, 256], 256, 'deconv_3_3')
        self.deconv_3_2 = c.deconv_layer(self.deconv_3_3, [3, 3, 256, 256], 256, 'deconv_3_2')
        self.deconv_3_1 = c.deconv_layer(self.deconv_3_2, [3, 3, 128, 256], 128, 'deconv_3_1')

        self.unpool_2 = c.unpool(self.deconv_3_1, self.pool_2_argmax)

        self.deconv_2_2 = c.deconv_layer(self.unpool_2, [3, 3, 128, 128], 128, 'deconv_2_2')
        self.deconv_2_1 = c.deconv_layer(self.deconv_2_2, [3, 3, 64, 128], 64, 'deconv_2_1')

        self.unpool_1 = c.unpool(self.deconv_2_1, self.pool_1_argmax)

        self.deconv_1_2 = c.deconv_layer(self.unpool_1, [3, 3, 64, 64], 64, 'deconv_1_2')
        self.deconv_1_1 = c.deconv_layer(self.deconv_1_2, [3, 3, 32, 64], 32, 'deconv_1_1')

        self.y_conv = c.deconv_layer(self.deconv_1_1, [1, 1, 3, 32], 3, 'score_1')
        self.y_soft = tf.nn.softmax(self.y_conv)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.cross_entropy_valid = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cross_entropy)

        #tf.summary.image('input x_image', self.x_image, 4)
        #tf.summary.image('y_prediction', self.y_conv, 4)
        #tf.summary.image('y_GT', self.y_, 4)
        #tf.summary.image('y_pred_softmax', self.y_soft, 4)
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        tf.summary.scalar('cross_entropy_valid', self.cross_entropy_valid)
        #tf.summary.scalar('learning rate', self.lr)
