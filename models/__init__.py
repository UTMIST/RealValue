import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.CNN_models.lenet import build_LeNet
from models.CNN_models.MiniVGGNet import MiniVGGNetModel
from models.CNN_models.RegNet import RegNet

def create_concat_network(Multi_Input):
    Final_Fully_Connected_Network = layers.Dense(32, activation = 'relu')(Multi_Input)
    Final_Fully_Connected_Network = layers.BatchNormalization()(Final_Fully_Connected_Network)
    #Final_Fully_Connected_Network = layers.Dense(16, activation = 'relu')(Final_Fully_Connected_Network)
    Final_Fully_Connected_Network = layers.Dense(1, activation = 'relu')(Final_Fully_Connected_Network)
    return Final_Fully_Connected_Network

def get_network(CNN_name, dense_layers, CNN_input_shape, input_shape=23) -> None:
    #create dense network dynamically based on input
    layer_list=[keras.Input(shape=(input_shape,))]
    for i in range(len(dense_layers)):
        name = "layer" + str(i+1)
        layer_list+=[layers.Dense(dense_layers[i], activation="relu", name=name)] #final output layer, no activation for next layer
    #layer_list+=[layers.Dense(1, activation="linear")]
    dense_model = keras.Sequential(layer_list)
    # print(dense_model.summary())
    #select the CNN network

    if CNN_name == 'LeNet':
        CNN_model = build_LeNet(input_shape=CNN_input_shape)

    elif CNN_name == 'MiniVGG':
        CNN_model = MiniVGGNetModel()
    elif CNN_name == 'VGG16':
        CNN_model = build_VGG16(input_shape=CNN_input_shape)
    elif CNN_name == 'ResNet':
        CNN_model = ResNet18()
    elif CNN_name == 'RegNet':
        CNN_model = RegNet(input_shape=CNN_input_shape)
    else:
        CNN_model = None
    print('True')

    # print(CNN_model.summary())
    return dense_model, CNN_model
