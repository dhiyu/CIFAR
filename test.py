import tensorflow as tf

with tf.variable_scope('v_scope'):
    with tf.name_scope('n_scope'):
        x = tf.Variable([1], name='x')
        y = tf.get_variable('g', shape=[1], dtype=tf.int32)
        z = x + y
#print(x, y, z)