from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0

def EfficientNet():
    """This code uses adapts the most up-to-date version of EfficientNet with NoisyStudent weights to a regression
    problem. Most of this code is adapted from the official keras documentation.

    Returns
    -------
    Model
        The keras model.
    """
    inputs = layers.Input(
        shape=(224, 224, 3)
    )  # input shapes of the images should always be 224x224x3 with EfficientNetB0

    # input_tensor 	Optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
    # input_shape 	Optional shape tuple, only to be specified if include_top is False. It should have exactly 3 inputs channels.

    # use the downloaded and converted newest EfficientNet wheights
    """TODO: figure out efficient net types (B0, B1, etc)"""
    # include_top=False means exclude the fully-connected layer at the top/end of the network.
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.4
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)

    # Match output shape of regnet
    outputs = layers.Dense(units=4, activation = 'relu', use_bias = True)(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")

    return model
