import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class VGGNet(object):
    def __init__(self, input_shape, num_classes, training=True):
        self.height, self.width = input_shape
        self.num_classes = num_classes
        self.training = training
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 3], name='input_holder')
        self.ground_truth = tf.placeholder(tf.float32, shape=[None, num_classes])

    def build(self):
        self.conv_1 = self.conv_layer('conv1', self.inputs, 64, [3, 3], 1)
        self.conv_2 = self.conv_layer('conv2', self.conv_1, 64, [3, 3], 1)
        self.pool1 = self.max_pooling('pool1', self.conv_2, [2, 2], 2)

        self.conv_3 = self.conv_layer('conv3', self.pool1, 128, [3, 3], 1)
        self.conv_4 = self.conv_layer('conv4', self.conv_3, 128, [3, 3], 1)
        self.pool2 = self.max_pooling('pool2', self.conv_4, [2, 2], 2)

        self.conv_5 = self.conv_layer('conv5', self.pool2, 256, [3, 3], 1)
        self.conv_6 = self.conv_layer('conv6', self.conv_5, 256, [3, 3], 1)
        self.conv_7 = self.conv_layer('conv7', self.conv_6, 256, [3, 3], 1)
        self.pool3 = self.max_pooling('pool3', self.conv_7, [2, 2], 2)

        self.conv_8 = self.conv_layer('conv8', self.pool3, 512, [3, 3], 1)
        self.conv_9 = self.conv_layer('conv9', self.conv_8, 512, [3, 3], 1)
        self.conv_10 = self.conv_layer('conv10', self.conv_9, 512, [3, 3], 1)
        self.pool4 = self.max_pooling('pool4', self.conv_10, [2, 2], 2)

        self.conv_11 = self.conv_layer('conv11', self.pool4, 512, [3, 3], 1)
        self.conv_12 = self.conv_layer('conv12', self.conv_11, 512, [3, 3], 1)
        self.conv_13 = self.conv_layer('conv13', self.conv_12, 512, [3, 3], 1)
        self.pool5 = self.max_pooling('pool5', self.conv_13, [2, 2], 2)

        self.reshape = tf.reshape(self.pool5, [-1, (self.height//32)*(self.width//32)*256])
        self.fc_14 = self.fc_layer('fc14', self.reshape, (self.height//32)*(self.width//32)*256, 1024)
        self.fc_15 = self.fc_layer('fc15', self.fc_14, 1024, 1024)
        self.fc_16 = self.fc_layer('fc16', self.fc_15, 1024, self.num_classes)
        self.softmax = tf.nn.softmax(tf.reshape(self.fc_16, [-1, self.num_classes]), name='prob')

        print(self.softmax)
        return self.softmax

    def conv_layer(self, name, inputs, filters, kernel_size, stride):
        channels = int(inputs.get_shape()[-1])
        print(inputs.shape, channels)
        with tf.variable_scope(name):
            # print(kernel)
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
                              strides=[1, stride, stride, 1], padding='SAME', name=name)

    def fc_layer(self, name, inputs, num_in, num_out):
        with tf.variable_scope(name):
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=self.training)
            biases = tf.get_variable('biases', shape=[num_out], trainable=self.training)
            fc = tf.nn.bias_add(tf.matmul(inputs, weights), biases)

        return fc

net = VGGNet([224, 224], 12)

