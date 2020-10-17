import tensorflow as tf
from tensorflow.keras import layers

def build_LeNet(input_shape=(240,240,3)):
    layer_list=[layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=input_shape),#(240,240,3)),#input_shape=(32,32,1)),
                layers.AveragePooling2D(),
                layers.BatchNormalization(),
                layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu'),
                layers.AveragePooling2D(),
                layers.BatchNormalization(),
                layers.Flatten(),
                layers.Dense(units=96, activation='relu'),
                layers.Dense(units=84, activation='relu'),
                layers.Dense(units=32, activation = 'softmax')
                ]
    model = tf.keras.Sequential(layer_list, name="LeNet")

    return model
