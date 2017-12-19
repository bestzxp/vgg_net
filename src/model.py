import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy

class VGGNet(object):
    def __init__(self, input_shape, num_classes, training=True):
        self.height, self.width = input_shape
        self.training = training
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 3], name='input_holder')
        self.outputs = tf.placeholder(tf.float32, shape=[None, num_classes])

    def build(self, inputs):
        self.conv_1 = self.conv_layer('conv1', inputs, 64, [3, 3], 1)
        self.conv_2 = self.conv_layer('conv2', self.conv_1, 64, [3, 3], 1)
        self.pool1 = self.max_pooling('pool1', self.conv_2, [2, 2], 2)

        self.conv_3 = self.conv_layer('conv3', self.pool1, 128, [3, 3], 1)
        self.conv_4 = self.conv_layer('conv4', self.conv_3, 128, [3, 3], 1)
        self.pool2 = self.max_pooling('pool2', self.conv_4, [2, 2], 2)
        pass

    def conv_layer(self, name, inputs, filters, kernel_size, stride):

        print(inputs.shape)
        channels = inputs.shape[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([kernel_size[0],
                                                                                kernel_size[1],
                                                                                channels,
                                                                                filters], stddev=0.1))
            biases = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[filters]))

            conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding='SAME', name=name)

            conv_batchnormed = slim.batch_norm(conv, center=False,
                                               scale=True, epsilon=1e-5, scope=name,
                                               is_training=self.training, updates_collections=None)
            conv_biases = tf.nn.bias_add(conv_batchnormed, biases)
            leaky = tf.maximum(conv_biases, conv_biases*0.1, name='leaky')

        return leaky

    def max_pooling(self, name, inputs, kernel_size, stride):
        return tf.nn.max_pool(inputs, [1, kernel_size[0], kernel_size[1], 1],
                              stride=[1, stride, stride, 1], padding='SAME', name=name)

    def fc_layer(self, name, inputs, num_in, num_out):
        with tf.variable_scope(name):
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=self.training)
            biases = tf.get_variable('biases', shape=[num_out], trainable=self.training)
            fc = tf.nn.bias_add(tf.matmul(inputs, weights), biases)

        return fc
