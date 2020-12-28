import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Dropout

def RegNet(input_shape=(64,64,3)):
    layer_list=[
                layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', use_bias=True, input_shape=input_shape, padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)),
                layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', use_bias=True, padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)),
                layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', use_bias=True, padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)),
                layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', use_bias=True, padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
                layers.Flatten(),
                #layers.Dense(units=128, activation = 'relu', use_bias = True, kernel_regularizer =tf.keras.regularizers.l2( l=0.01)),# not sure about the input shape
                layers.Dense(units=128, activation = 'relu', use_bias = True),
                layers.BatchNormalization(),
                layers.Dropout(0.45),
                layers.Dense(units=32, activation = 'linear', use_bias = True),
                #layers.Dense(units=1, activation = 'linear')
                ]
    model = tf.keras.Sequential(layer_list, name="RegNet")
    print(model.summary())
    return model
