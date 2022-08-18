import os.path

import torch
# import torchvision
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets('./mnist_data/', one_hot=True)
mnist = tf.keras.datasets.mnist
# 加载手写数字和标签
(train_x, train_y), (test_x, test_y) = mnist.load_data()
print(train_y.shape, train_x.shape, test_x.shape, test_y.shape)

# 转化为指定数据类型
x_train, x_test = tf.cast(train_x / 255.0, tf.float32), tf.cast(test_x / 255.0, tf.float32)
y_train, y_test = tf.cast(train_y, tf.float32), tf.cast(test_y, tf.float32)

num_inputs = 784
num_outputs = 10
num_hiddens = 500


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, std=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    w1 = get_weight(num_inputs, num_hiddens)
    b1 = get_bias([num_hiddens])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight(num_hiddens, num_outputs)
    b2 = get_bias([num_outputs])
    y = tf.matmul(y1, w2) + b2
    return y


def sgd(params, lr):
    with torch.no_grad():
        for p in params:
            p -= p.grad * lr
            p.grad_zero_()


regularizer = 0.0001
batch_size = 200
lr = 0.1
num_epochs = 50000
model_save_path = './Lwarn_Model/'
model_name = 'mnist_model'


def backward(mnist):
    y = forward(x_train, regularizer)
    global_step = tf.Variable(0, trainable=False)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    ce_mean = tf.reduce_mean(ce)
    loss = ce_mean + tf.add_n(tf.get_collection('losses'))
    train_step = sgd(lr=lr).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)

    ema = tf.train.Exponentiall

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(num_epochs):
            xs, ys = mnist.train.next_batch(batch_size)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print(f'After {step} training step(s),loss on training batch is {loss_value}')
                saver.save(sess, os.path.join(model_save_path, model_name), global_step=step)
