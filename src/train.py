import tensorflow as tf
from src.model import VGGNet
from src.data_loader import DataLoader
net = VGGNet([224, 224], 12)
net.build()
loss = net.loss()
# print(tf.global_variables())
loader = DataLoader()
# 166.111.83.102

pred_label = tf.argmax(net.softmax)
truth_label = tf.argmax(net.ground_truth)
correct_prediction = tf.equal(pred_label, truth_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# learning_rate = tf.train.exponential_decay(0.00001, global_step, 15,
#             0.9, name='learning_rate')
optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

batch = 32
batch_num = loader.images_urls.shape[0] // batch
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, '../ckpt/model.ckpt')
    for i in range(100000):
        total_loss = 0
        total_ac = 0
        for _ in range(batch_num):
            res = loader.get_batch_data(batch)
            feed_dicts = {net.inputs: res[0], net.ground_truth: res[1]}
            # sess.run(optimizer, feed_dict=feed_dicts)
            _, l, a = sess.run([optimizer, loss, accuracy], feed_dict=feed_dicts)
            print('    ', l, a)
            total_loss += l
            total_ac += a
        print(i, total_loss, total_ac / batch_num)
        loader.shuffle()
        if i % 5 == 0:
            saver.save(sess, '../ckpt/model.ckpt', i)
