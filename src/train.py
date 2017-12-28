import tensorflow as tf
from src.model import VGGNet
from src.data_loader import DataLoader
net = VGGNet([32, 32], 12)
net.build()
loader = DataLoader()
# 166.111.83.102
optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
