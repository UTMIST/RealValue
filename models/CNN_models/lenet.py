import tensorflow as tf

class LeNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(LeNet, self).__init__()
        self.conv_layer_1 = tf.keras.layers.Conv2D(
                filters=6,
                kernel_size=(5, 5),
                input_shape=(28, 28, 1),
                padding='valid',
                activation=tf.nn.relu
                )
        self.pool_layer_1 = tf.keras.layers.MaxPooling2D(padding='same')
        self.conv_layer_2 = tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=(5, 5),
                padding='valid',
                activation=tf.nn.relu
                )
        self.pool_layer_2 = tf.keras.layers.MaxPooling2D(padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.fc_layer_1 = tf.keras.layers.Dense(
                units=120,
                activation=tf.nn.relu
                )
        self.fc_layer_2 = tf.keras.layers.Dense(
                units=84,
                activation=tf.nn.relu
                )
        self.output_layer = tf.keras.layers.Dense(
                units=kwargs['num_classes'],
                activation=tf.nn.softmax
                )

    @tf.function
    def call(self, features):
        activation = self.conv_layer_1(features)
        activation = self.pool_layer_1(activation)
        activation = self.conv_layer_2(activation)
        activation = self.pool_layer_2(activation)
        activation = self.flatten(activation)
        activation = self.fc_layer_1(activation)
        activation = self.fc_layer_2(activation)
        output = self.output_layer(activation)
        return output
