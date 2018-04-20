import tensorflow as tf
import sys
from model import VGGNet
from data_loader import DataLoader
net = VGGNet([224, 224], 128)
net.build()
loss = net.loss()
# print(tf.global_variables())
ckpt_path = '../ckpt/model.ckpt-0'

loader = DataLoader()

sess = tf.Session()
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

ls = tf.summary.scalar('loss', loss)

train_writer = tf.summary.FileWriter('../log_train', sess.graph)
valid_writer = tf.summary.FileWriter('../log_valid', sess.graph)

batch = 32
batch_num = loader.images_urls.shape[0] // batch
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.7
valid_batch_num = loader.valid_urls.shape[0] // batch

if ckpt_path:
    saver.restore(sess, ckpt_path)
else:
    sess.run(tf.global_variables_initializer())

global_step = 0
valid_step = 0
for i in range(100000):

    total_loss = 0
    for idx in range(valid_batch_num):
        valid_step += 1
        res = loader.get_valid_batch_data(batch)
        feed_dicts = {net.inputs: res[0], net.ground_truth: res[1]}
        # sess.run(optimizer, feed_dict=feed_dicts)
        ls_, l = sess.run([ls, loss], feed_dict=feed_dicts)
        total_loss += l
        valid_writer.add_summary(ls_, valid_step)
        sys.stdout.write("\r-valid epoch:%3d, idx:%4d, loss: %0.6f" % (i, idx, l))
    loader.valid_cursor = 0
    print("\nepoch:{}, valid avg_loss:{}".format(i, total_loss / valid_batch_num))


    total_loss = 0
    for idx in range(batch_num):
        global_step += 1
        res = loader.get_batch_data(batch)
        feed_dicts = {net.inputs: res[0], net.ground_truth: res[1]}
        # sess.run(optimizer, feed_dict=feed_dicts)
        _, ls_, l = sess.run([optimizer, ls, loss], feed_dict=feed_dicts)
        total_loss+=l
        train_writer.add_summary(ls_, global_step)
        sys.stdout.write("\r-train epoch:%3d, idx:%4d, loss: %0.6f" % (i, idx, l))
    print("\nepoch:{}, train avg_loss:{}".format(i, total_loss/batch_num))
    saver.save(sess, '../ckpt/model_{}.ckpt'.format(i))


    loader.shuffle()

