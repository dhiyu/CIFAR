import tensorflow as tf
import numpy as np


# 神经网络前向传播模块


input_node = 1024
output_node = 10

image_size = 32
# 三通道图像
num_channels = 3
num_labels = 10

conv1_deep = 32
conv1_size = 5

conv2_deep = 64
conv2_size = 5

fc_size = 512


def inference(input_tensor, train, regularizer):
    # 输入32*32*3矩阵
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(name="weight", shape=[conv1_size, conv1_size, num_channels, conv1_deep], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(name="bias", shape=[conv1_deep], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    # 输出32*32*32 矩阵
    with tf.name_scope('layer2-pool1'):
            pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 输出16*16*32矩阵
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(name="weight", shape=[conv2_size, conv2_size, conv1_deep, conv2_deep], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(name="bias", shape=[conv2_deep], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    # 输出8*8*64矩阵
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    #全连接层

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, fc_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer !=None:
            tf.add_to_collection("losses", regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [fc_size], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights)+fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

        with tf.variable_scope('layer6-fc2'):
            fc2_weights = tf.get_variable("weight", [fc_size, num_labels], initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None:
                tf.add_to_collection("losses", regularizer(fc2_weights))
            fc2_biases = tf.get_variable("bias", [num_labels], initializer=tf.constant_initializer(0.1))
            logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    return logit