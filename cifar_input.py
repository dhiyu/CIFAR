import numpy as np
import tensorflow as tf

data_route = 'cifar-10-batches-py'


def read_data_batch(batchsize=10):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([data_route+'/data_batch.tfrecords'])
    _, serialized_example = reader.read_up_to(filename_queue, num_records=batchsize)
    features = tf.parse_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64), 'img_raw': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [batchsize, 32,32,3])
    label = tf.cast(features['label'], tf.int32)
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return image


def read_test_batch(batchsize=10):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([data_route + '/test_batch.tfrecords'])
    _, serialized_example = reader.read_up_to(filename_queue, num_records=batchsize)
    features = tf.parse_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64), 'img_raw': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.float32)
    image = tf.cast(image, tf.float32)
    # 动态分配显存
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    print(sess.run([image, label]))
    '''for i in range(batchsize):
        print(sess.run([image, label])[0])'''


print(read_test_batch(batchsize=10))