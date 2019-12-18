import tensorflow as tf


def lenet(args, x, size):
    shape = args.studentargs

    output = tf.reshape(x, shape=[-1, size])
    layer_sizes = [int(arg) for arg in shape[1:]]
    for i, layer_size in enumerate(layer_sizes[:-1]):
        output = tf.layers.dense(output,
                                 layer_size,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                     scale=0.01),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(
                                     scale=0.01),
                                 activation=tf.nn.relu,
                                 name='student_dense%d_layer' % i)
    output = tf.layers.dense(output,
                             layer_sizes[-1],
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                 scale=0.01),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(
                                 scale=0.01),
                             name='student_output_layer')
    return tf.identity(output, name='student_model_output')
