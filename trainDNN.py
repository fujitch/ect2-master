# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
import numpy as np
import random

dataset = pickle.load(open('dataset.pickle', 'rb'))
dataset = dataset[:6318]

random.shuffle(dataset)

datasetMatrix = np.zeros((6318, 99))
for i in range(len(dataset)):
    datasetMatrix[i, :] = dataset[i]

dataset = datasetMatrix

batch_size = 100
training_epochs = 10000
display_epochs = 100

"""
# RMSの計算
def calRms(data):
    square = np.power(data,2)
    rms = math.sqrt(sum(square)/len(data))
    return rms
def calRms2D(data):
    square = np.power(data,2)
    rms = math.sqrt(sum(sum(square))/len(data))
    return rms

otherMatrix = np.zeros((len(dataset), 93))

for i in range(len(dataset)):
    otherMatrix[i, :] = dataset[i][6:]
"""

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
datasetTrain = dataset[:6000, :]
datasetTest = dataset[6000:, :]

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

"""
# バッチ作成
def make_batch(batch_size):
    batch = np.zeros((batch_size, 97))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 2))
    output = np.array(output, dtype=np.float32)
    for i in range(batch_size):
        index = random.randint(0, len(dataset) - 1)
        batch[i, 0] = dataset[index][2] / coilRMS
        batch[i, 1] = dataset[index][3] / liftRMS
        batch[i, 2] = dataset[index][4] / frequencyRMS
        batch[i, 3] = dataset[index][5] / conductivityRMS
        batch[i, 4:] = dataset[index][6:] / otherRMS
        output[i, 0] = dataset[index][0]
        output[i, 1] = dataset[index][1] / depthRMS
    return batch, output

def make_batch_test(batch_size):
    batch = np.zeros((batch_size, 97))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 2))
    output = np.array(output, dtype=np.float32)
    for i in range(batch_size):
        index = random.randint(0, len(datasetTest) - 1)
        batch[i, 0] = datasetTest[index][2] / coilRMS
        batch[i, 1] = datasetTest[index][3] / liftRMS
        batch[i, 2] = datasetTest[index][4] / frequencyRMS
        batch[i, 3] = datasetTest[index][5] / conductivityRMS
        batch[i, 4:] = datasetTest[index][6:] / otherRMS
        output[i, 0] = datasetTest[index][0]
        output[i, 1] = datasetTest[index][1] / depthRMS
    return batch, output
"""

W_fc1 = weight_variable([97, 1024])
b_fc1 = bias_variable([1024])
h_flat = tf.reshape(x, [-1, 97])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 256])
b_fc2 = bias_variable([256])
h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([256, 64])
b_fc3 = bias_variable([64])
h_fc3 = tf.nn.sigmoid(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

W_fc4 = weight_variable([64, 2])
b_fc4 = bias_variable([2])
y_out = tf.matmul(h_fc3_drop, W_fc4) + b_fc4

each_square = tf.square(y_ - y_out)
loss = tf.reduce_mean(each_square)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()

saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())
# saver.restore(sess, "./20181031model")
for i in range(training_epochs):
    for k in range(0, len(datasetTrain), batch_size):
        batch = datasetTrain[k:k+batch_size, 2:]
        output = datasetTrain[k:k+batch_size, :2]
        train_step.run(session=sess, feed_dict={x: batch, y_: output, keep_prob: 0.5})
    
    if i%display_epochs == 0:
        batch = datasetTrain[:, 2:]
        output = datasetTrain[:, :2]
        train_loss = loss.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
        batch = datasetTest[:, 2:]
        output = datasetTest[:, :2]
        test_loss = loss.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
        print(str(i) + "epochs_finished!")
        print("train_loss===" + str(train_loss))
        print("test_loss===" + str(test_loss))

saver.save(sess, "./20181101model")
sess.close()