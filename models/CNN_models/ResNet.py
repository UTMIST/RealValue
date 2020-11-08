import tensorflow as tf
from tensorflow.keras import layers


def _residual_block_first(self, x, out_channel, strides, name="unit"):
    in_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)

        # Shortcut connection
        if in_channel == out_channel:
            if strides == 1:
                shortcut = tf.identity(x)
            else:
                shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
        else:
            shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
        # Residual
        x = self._conv(x, 3, out_channel, strides, name='conv_1')
        x = self._bn(x, name='bn_1')
        x = self._relu(x, name='relu_1')
        x = self._conv(x, 3, out_channel, 1, name='conv_2')
        x = self._bn(x, name='bn_2')
        # Merge
        x = x + shortcut
        x = self._relu(x, name='relu_2')
    return x


def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        shortcut = x
        # Residual
        x = self._conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
        x = self._bn(x, name='bn_1')
        x = self._relu(x, name='relu_1')
        x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
        x = self._bn(x, name='bn_2')

        x = x + shortcut
        x = self._relu(x, name='relu_2')
    return x
