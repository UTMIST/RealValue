import tensorflow as tf

class SimpleDenseNet(tf.keras.Model):
    def __init__(self, dim,**kwargs):
        super(SimpleDenseNet, self).__init__()
        self.dense_layer_1 = tf.keras.layers.Dense(
                units=8,
                input_dim=dim,
                activation='relu'
        )
        self.dense_layer_2 = tf.keras.layers.Dense(
                units=4,
                activation='relu'
        )

    @tf.function
    def call(self, features):
        activation = self.dense_layer_1(features)
        output = self.dense_layer_2(activation)

        return output
