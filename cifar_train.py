import os
import tensorflow as tf
import cifar_inference
import cifar_input


batch_size = 100
learning_rate_base = 0.8
learning_rate_decay = 0.99
regularaztion_rate = 0.0001
training_steps = 100000
moving_average_decay = 0.99

model_save_path = 'model_save/'
model_name = "cifar10_model.ckpt"


def train(input_tensor):
    # 定义输入输出占位:
    x = tf.placeholder(tf.float32, [None, cifar_inference.input_node], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, cifar_inference.output_node], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(regularaztion_rate)
    y = cifar_inference.inference(input_tensor, 1, regularizer)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables)  # teat_point_1
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, lebels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('loss'))
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 600, learning_rate_decay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op('train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        start_position = 0
        for i in range(training_steps):
            if start_position >=60000:
                start_position -=60000
            xs, ys = cifar_input.read_next_batch(start_position=start_position, batch_size=batch_size)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            start_position += batch_size

            if i%10 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)


def main():
    cifar = cifar_input.read_img(0)
    train(cifar)

main()
