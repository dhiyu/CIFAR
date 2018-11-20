import os
import tensorflow as tf
import cifar_inference
import numpy as np

batch_size = 10
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularaztion_rate = 0.0001
training_steps = 1000000
moving_average_decay = 0.9999

model_save_path = 'model_save/'
model_name = "cifar10_model.ckpt"
data_route = 'cifar-10-batches-py'


def train():
    # 动态分配显存
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([data_route + '/data_batch.tfrecords'])
    _, serialized_example = reader.read_up_to(filename_queue, num_records=batch_size)
    features = tf.parse_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64), 'img_raw': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [batch_size, 32, 32, 3])
    '''# 随机水平翻转
    distorted_image = tf.image.random_flip_left_right(image)
    # 随机调整亮度
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    # 随机调整对比度
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    # 对图像进行白化操作，即像素值转为零均值单位方差
    image = tf.image.per_image_standardization(distorted_image)'''
    label = tf.cast(features['label'], tf.uint8)
    label = tf.one_hot(label, batch_size, 1, 0)
    label = tf.reshape(label, [batch_size, 10])
    label = tf.cast(label, tf.float32)
    # label = np.array(label)
    image = tf.cast(image, tf.float32)
    coord = tf.train.Coordinator()
    print(label)

    # 定义输入输出占位:
    x = tf.placeholder(tf.float32, [None, cifar_inference.image_size, cifar_inference.image_size, cifar_inference.num_channels], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, cifar_inference.output_node], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(regularaztion_rate)
    y = cifar_inference.inference(image, 1, regularizer)
    # print(y)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())  # teat_point_1
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # print(cross_entropy_mean)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # print(loss)
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 600, learning_rate_decay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op('train')

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        queue_runner = tf.train.start_queue_runners(sess=sess, coord=coord)
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter("I:/TensorBoard/test", sess.graph)
        writer.close()
        for i in range(training_steps):
            img = sess.run(image)
            lab = sess.run(label)
            xs, ys = img, lab
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print(type(loss_value))
                print("After %d training step(s), loss on training batch is %f." % (step, loss_value))
                saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)
        coord.request_stop()
        coord.join(queue_runner)


train()