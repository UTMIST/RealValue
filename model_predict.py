import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from step1 import return_splits
import time
import yaml
import csv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import global_vars as GLOBALS
import time
import numpy as np
import random
import models
from models import get_network
from split_and_augment_dataset import split_and_augment_train_dataset
from contextlib import contextmanager
from pipeline import initialize_hyper, create_data

path_to_config="config.yaml"
GLOBALS.CONFIG = initialize_hyper(path_to_config)
if GLOBALS.CONFIG is None:
    print("error in initialize_hyper")
    sys.exit(1)

# load dense mode

# load data

# print("start initializing dataset")
# initialize_datasets()
# print("finished initializing dataset")
data_dict = create_data()

GLOBALS.CONFIG['input_shape'] = data_dict['validation_stats'].shape[1]
Dense_NN, _ = get_network(GLOBALS.CONFIG['CNN_model'], dense_layers=GLOBALS.CONFIG['dense_model'], CNN_input_shape=GLOBALS.CONFIG['CNN_input_shape'], input_shape=GLOBALS.CONFIG['input_shape'])

print(Dense_NN.summary())

# load weights
# path = "Output_Files\dense_nn\output_folder_sequential_0.1_None_raw_dataset_3\model_weights\model_weights.h5"
# path = "Output_Files\dense_output_layer_1\output_folder_functional_3_0.001_None_raw_dataset_50\model_weights\model_weights.h5"
# path = "Output_Files\dense_output_layer_1_attempt2\output_folder_functional_3_0.001_None_raw_dataset_3\model_weights\model_weights.h5"
# path = "Output_Files\dense_output_layer_1_attempt2\output_folder_functional_5_0.002_None_raw_dataset_3\model_weights\model_weights.h5"
# path = "Output_Files\dense_nn\output_folder_sequential_0.1_None_raw_dataset_3\model_weights\model_weights.h5"

# mape of 64.7
path = "Output_Files\dense_nn\output_folder_sequential_1e-04_None_raw_dataset_250\model_weights\model_weights.h5"

Dense_NN.load_weights(path)




#sample_house_tensor = tf.constant([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0., 0.,0.,0.,0.,0.38762439])
#print(sample_house_tensor.shape)
# sample_house_tensor = tf.constant([0.,0.,1.,0.,0.,0.
#  0.,0.,0.,0.,0.,0.
#  0.,0.,0.,1.,0.,0.
#  0.,0.,0.,0.,0.38762439]

sample_house_data = data_dict['test_stats'][0]
print(data_dict['test_min_max'])
# {'price_min': 50000.0, 'sqft_min': 840.0, 'price_max': 2150000.0, 'sqft_max': 9583.0}

print("test stats first value")
print(data_dict['test_stats'][0])

sample_house_tensor_reshaped = tf.reshape(sample_house_data,(-1,3))

print(Dense_NN.summary())

# price = Dense_NN.predict(sample_house_tensor)
price = Dense_NN.predict(sample_house_tensor_reshaped)




# evaluate
# print(price.numpy())
print("Price (normalized)")
print(price)

print("Price prediction")
print(price * data_dict['test_min_max']["price_max"])

print("Actual price")
print(data_dict['test_prices'][0])
