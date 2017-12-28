import tensorflow as tf
from src.model import VGGNet
from src.data_loader import DataLoader
net = VGGNet([320, 320], 12)
net.build()
loss = net.loss()
# print(tf.global_variables())
loader = DataLoader()
# 166.111.83.102
optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

pred_label = tf.argmax(net.softmax)
truth_label = tf.argmax(net.ground_truth)
correct_prediction = tf.equal(pred_label, truth_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        res = loader.get_batch_data(8)
        feed_dicts = {net.inputs: res[0], net.ground_truth: res[1]}
        # sess.run(optimizer, feed_dict=feed_dicts)
        _, l, a = sess.run([optimizer, loss, accuracy], feed_dict=feed_dicts)
        print(l, a)
