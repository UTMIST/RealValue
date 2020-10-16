import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from models.CNN_models.vgg import VGG16
#from models.CNN_models.ResNet18 import ResNet18
from models.CNN_models.lenet import LeNet
#from models.CNN_models.name_of_file_import name_of_class

def get_network(CNN_name, dense_layers) -> None:
    #create dense network dynamically based on input 
    dense_model = keras.Sequential(
        [
            for i in range(len(dense_layers)-1): #length of dense_layers is how many layers there will be
                name = "layer" + str(i+1)
                layers.Dense(dense_layers[i], activation="relu", name=name),
            name = "layer" + str(len(dense_layers))
            layers.Dense(dense_layers[len(dense_layers)-1], name=name), #final output layer, no activation for next layer
        ]
    )
    #select the CNN network
    if name == 'LeNet':
        CNN_model = LeNet()
    elif name == 'VGG16':
        CNN_model = VGG16()
    elif name == 'ResNet':
        CNN_model = ResNet18()
    else:
        CNN_model = None

    return dense_model, CNN_model
