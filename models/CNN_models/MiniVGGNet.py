import tensorflow as tf

class MiniVGGNetModel(tf.keras.Model):
    def __init__(self, classes=4, chanDim=-1):
        # call the parent constructor
        super(MiniVGGNetModel, self).__init__()
        # initialize the layers in the first (CONV => RELU) * 2 => POOL
        # layer set
        self.conv1A = tf.keras.layers.Conv2D(32, (3, 3), padding="same")
        self.act1A = tf.keras.layers.Activation("relu")
        self.bn1A = tf.keras.layers.BatchNormalization(axis=chanDim)
        self.conv1B = tf.keras.layers.Conv2D(32, (3, 3), padding="same")
        self.act1B = tf.keras.layers.Activation("relu")
        self.bn1B = tf.keras.layers.BatchNormalization(axis=chanDim)
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        # initialize the layers in the second (CONV => RELU) * 2 => POOL
        # layer set
        self.conv2A = tf.keras.layers.Conv2D(32, (3, 3), padding="same")
        self.act2A = tf.keras.layers.Activation("relu")
        self.bn2A = tf.keras.layers.BatchNormalization(axis=chanDim)
        self.conv2B = tf.keras.layers.Conv2D(32, (3, 3), padding="same")
        self.act2B = tf.keras.layers.Activation("relu")
        self.bn2B = tf.keras.layers.BatchNormalization(axis=chanDim)
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        # initialize the layers in our fully-connected layer set
        self.flatten = tf.keras.layers.Flatten()
        self.dense3 = tf.keras.layers.Dense(512)
        self.act3 = tf.keras.layers.Activation("relu")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.do3 = tf.keras.layers.Dropout(0.5)
        # initialize the layers in the softmax classifier layer set
        self.dense4 = tf.keras.layers.Dense(classes)
        self.softmax = tf.keras.layers.Activation("softmax")

    def call(self, inputs):
        # build the first (CONV => RELU) * 2 => POOL layer set
        x = self.conv1A(inputs)
        x = self.act1A(x)
        x = self.bn1A(x)
        x = self.conv1B(x)
        x = self.act1B(x)
        x = self.bn1B(x)
        x = self.pool1(x)
        # build the second (CONV => RELU) * 2 => POOL layer set
        x = self.conv2A(x)
        x = self.act2A(x)
        x = self.bn2A(x)
        x = self.conv2B(x)
        x = self.act2B(x)
        x = self.bn2B(x)
        x = self.pool2(x)
        # build our FC layer set
        x = self.flatten(x)
        x = self.dense3(x)
        x = self.act3(x)
        x = self.bn3(x)
        x = self.do3(x)
        # build the softmax classifier
        x = self.dense4(x)
        x = self.softmax(x)
        # return the constructed model
        return x

    def model(self):
        x = tf.keras.Input(shape=(24, 24, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
