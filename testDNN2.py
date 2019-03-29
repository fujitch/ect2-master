# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
import numpy as np
import random

dataset = pickle.load(open('dataset.pickle', 'rb'))

random.shuffle(dataset)

datasetMatrix = np.zeros((25272, 99))
for i in range(len(dataset)):
    datasetMatrix[i, :] = dataset[i]

dataset = datasetMatrix

batch_size = 100
training_epochs = 10000
display_epochs = 100

datasetDummy = []
for data in dataset:
    if data[2] == 7:
        if data[4] == 400:
            datasetDummy.append(data)
dataset = datasetDummy

random.shuffle(dataset)
datasetMatrix = np.zeros((len(dataset), dataset[0].shape[0]))
for i in range(len(dataset)):
    datasetMatrix[i, :] = dataset[i]
dataset = datasetMatrix
dataset = dataset[:300, :]

coilRMS = 7
liftRMS = 5
frequencyRMS = 400
conductivityRMS = 100
widthRMS = 0.5
depthRMS = 10

# 規格化
dataset[:, 0] /= widthRMS
dataset[:, 1] /= depthRMS
dataset[:, 2] /= coilRMS
dataset[:, 3] /= liftRMS
dataset[:, 4] /= frequencyRMS
dataset[:, 5] /= conductivityRMS
dataset[:, 6:] -= np.min(dataset[:, 6:])
dataset[:, 6:] /= np.max(dataset[:, 6:])

# 型変換
dataset = np.array(dataset, dtype=np.float32)

# datasetを分ける
"""
datasetTrain = dataset[:25000, :]
datasetTest = dataset[25000:, :]
"""

tf.reset_default_graph()

x = tf.placeholder("float", shape=[None, 97])
y_ = tf.placeholder("float", shape=[None, 2])

# 荷重作成
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアス作成
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

W_fc1 = weight_variable([97, 2048])
b_fc1 = bias_variable([2048])
h_flat = tf.reshape(x, [-1, 97])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([2048, 2048])
b_fc2 = bias_variable([2048])
h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([2048, 256])
b_fc3 = bias_variable([256])
h_fc3 = tf.nn.sigmoid(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

W_fc4 = weight_variable([256, 2])
b_fc4 = bias_variable([2])
y_out = tf.matmul(h_fc3_drop, W_fc4) + b_fc4

each_square = tf.square(y_ - y_out)
loss = tf.reduce_mean(each_square)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()

saver = tf.train.Saver()

# sess.run(tf.initialize_all_variables())
saver.restore(sess, "./20181101model")

batch = dataset[:, 2:]
output = dataset[:, :2]
out = y_out.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})

sess.close()
