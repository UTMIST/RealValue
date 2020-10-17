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
# from models import get_network
import models
from models import get_network


# from models import get_network
#from models.CNN_models.lenet import LeNet
#from models.dense_models.simple_densenet import SimpleDenseNet


def initialize_hyper(path_to_config):
    '''
    Reads config.yaml to set hyperparameters
    '''
    with open(path_to_config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            return None

def initialize_datasets():
    # augments datasets if it doesn't already exist
    # loads datasets otherwise
    pass

def train(path_to_config='config.yaml'):
    '''
    Inputs: The config.yaml file
    Output: Training history from model.fit, results from model.evaluate, the model itself

    The goal of this function is to conduct training.
    The process occurs in the following steps.

    1. Initialize all hyperparameters using the initialize_hyper function detailed above.
    2. Initialize/create augmented datasets (if it doesn't exist already)
    - This initialization will be done in a function called initialize_datasets (done in Step 1, so dw)
    - See Step 1 Workflow Notes for outputs of that function.
    3. Conduct the training process as follows:
    - model = Model(inputs = [Dense_NN.input , CNN.input], outputs = Final_Fully_Connected_Network)
      - The sub-bullets below are an example of how the model should be initialized. But note that the 3 lines below are done in another file.
      - Multi_Input = concatenate([Dense_NN.output, CNN.output])
      - Final_Fully_Connected_Network = Dense((whatever we want), activation = 'relu')(Multi_Input)
      - Final_Fully_Connected_Network = Dense(1)(Final_Fully_Connected_Network)
    - model.compile (with appropriate hyperparameters)
    - history = model.fit (with hyperparameters & training/validation results from 2)
    - results = model.evaluate (with hyperparameters & test results from 2)

    Refer to https://github.com/omarsayed7/House-price-estimation-from-visual-and-textual-features/blob/master/visual_textual_2.py for a sample implementation.

    #TO-DO: Recreate the models/__init__.py from the AdaS repository for our purposes.
    '''
    config = initialize_hyper(path_to_config)
    print(config)
    if config is None:
        print("error in initialize_hyper")
        sys.exit(1)
    GLOBALS.CONFIG=config

    #train_images, train_stats, train_prices, validation_images, validation_stats, validation_prices, \
        #test_images, test_stats, test_prices = return_splits( ... )
    directory='raw_dataset'
    data_dict = return_splits(directory, GLOBALS.CONFIG['train_val_test_split'])

    print('Train Images:',data_dict['train_images'].shape)
    print('Train Stats:',data_dict['train_stats'].shape)
    print('Train Prices:',data_dict['train_prices'].shape)
    print('Validation Images:',data_dict['validation_images'].shape)
    print('Validation Stats:',data_dict['validation_stats'].shape)
    print('Validation Prices:',data_dict['validation_prices'].shape)
    print('Test Images:',data_dict['test_images'].shape)
    print('Test Validation:',data_dict['test_stats'].shape)
    print('Test Prices:',data_dict['test_prices'].shape)

    CNN_type = config['CNN_model']
    Dense_NN, CNN = get_network(CNN_type, dense_layers=config['dense_model'], CNN_input_shape=config['CNN_input_shape'])
    # Dense_NN = SimpleDenseNet(train_stats.shape[1])
    Multi_Input = tf.keras.layers.concatenate([Dense_NN.output, CNN.output])

    Final_Fully_Connected_Network = tf.keras.layers.Dense(32, activation = 'relu')(Multi_Input)
    Final_Fully_Connected_Network = tf.keras.layers.Dense(1, activation = 'sigmoid')(Final_Fully_Connected_Network)

    model = Model(inputs = [Dense_NN.input , CNN.input], outputs = Final_Fully_Connected_Network)

    optimizer_functions={'Adam':keras.optimizers.Adam}
    optimizer=optimizer_functions[config['optimizer']](lr= config['learning_rate'])

    model.compile(optimizer=optimizer, loss = config['loss_function'],
        metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
    # history = model.fit([train_stats,train_images], train_prices, validation_split = config['validation_split'],
    #         epochs = config['epochs'],
    #         batch_size = config['batch_size'],
    #         callbacks= [tensorboard]) #not sure if we have the tensorfboard callback
    print("model.fit Debugging Info")
    print(data_dict["train_stats"][0].shape)
    print(data_dict["train_stats"][0].dtype)
    print(data_dict["train_stats"][0])
    history = model.fit(x=[data_dict["train_stats"],data_dict['train_images']], y=data_dict['train_prices'], validation_data=([data_dict["validation_stats"],data_dict['validation_images']], data_dict['validation_prices']),
            epochs = config['number_of_epochs'],
            batch_size = config['mini_batch_size'])

    #I'm not sure how we are incorporating the validation dataset into our training code?
    #preds = model.predict([data_dict['test_stats'],data_dict['test_images']])
    #print(preds)


    # results = model.evaluate (with hyperparameters & test results from 2)
    # btw, I also added these lines below for some other metrics I need for plotting
    # i guess we can ask Arsh where the model evaluation code should be ok
    results = model.evaluate([data_dict['test_stats'],data_dict['test_images']], data_dict['test_prices'], batch_size=config['mini_batch_size'])
    evaluation_results = dict(zip(model.metrics_names, results))

    return model, history, results

# train()

def save_model(model, model_dir):
    try:
        path = os.path.join(model_dir, "mode_weights.h5")
        model.save_weights(path)
    except:
        print("error saving model weights")
        return False
    return True

def plot(x, y, xlabel, ylabel, title, save=False, filename=None):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save:
        plt.savefig(filename)

    plt.show()

def save_dict_to_csv(dict, csv_file_path, fieldnames_header, start_row_num_from_1):
    # assumes a dictionary of lists like history.history

    with open(csv_file_path, 'w+', newline='') as csv_file:
        writer = csv.writer(csv_file)

        #
        writer.writerow(fieldnames_header)

        for row_number in range(len(list(dict.values())[0])):
            list_value = [list[row_number] for list in dict.values()]
            if start_row_num_from_1:
                writer.writerow([row_number + 1] + list_value)
            else:
                writer.writerow([row_number] + list_value)

def convert_csv_to_dict(csv_file_path):
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)

        dict = {}
        for row in reader:
            for column, value in row.items():
                dict.setdefault(column, []).append(value)

    return dict

def process_outputs(model, history_dict, results, scheduler, dataset, number_of_epochs):
    '''
    Inputs: History, results and model
    Output: Nothing, everything happens as function runs.

    The goal of this function is to create the correct output files from the training process.
    The process occurs in the following steps:

    1. Tap into the dictionary that is history.history
    - comes from model.fit
    2. Create graphs for accuracy and mean average percentage error using matplotlib
    - training and validation
    - comes from model.evaluate()
    - Do the above for the validation results as well.
    3. Also store the accuracy/loss statistics in an Excel file.
    4. Store model weights using model.save_weights
    5. Store final training, validation and test results (accuracy, error, network size) in a separate Excel file.

    Altogether, the corresponding output file structure should look as follows:

    output_folder_(modelname)_(learningrate)_(scheduler)_(dataset)_(numberofepochs)
    --> model_weights
      -- model_weights.h5
    --> graphs
      -- train_accuracy_graph.png
      -- validation_accuracy_graph.png
      -- train_loss_graph.png
      -- validation_loss.png
    --> stats_files
      -- train_accuracy.csv
      -- validation_accuracy.csv
      -- train_loss.csv
      -- validation_loss.csv
    --> results_files
      -- final_results.csv
    '''

    # Create output directory and subdirectory paths for model weights and results
    model_name = model.name
    learning_rate = tf.keras.backend.eval(model.optimizer.lr)

    output_folder_name = "output_folder_%s_%s_%s_%s_%s" % (model_name, learning_rate, scheduler, dataset, number_of_epochs)
    output_dir = os.path.join(os.path.dirname(__file__), output_folder_name)
    model_weights_dir = os.path.join(output_dir, "model_weights")
    graphs_dir = os.path.join(output_dir, "graphs")
    stats_dir = os.path.join(output_dir, "stats")
    results_dir = os.path.join(output_dir, "results_files")

    # Create output directories storing all results and model weights
    if not os.path.exists(os.path.join(os.path.dirname(__file__), output_dir)):
        os.makedirs(output_dir)
        os.makedirs(model_weights_dir)
        os.makedirs(graphs_dir)
        os.makedirs(stats_dir)
        os.makedirs(results_dir)

    # Save training history (loss, sparse_categorical_accuracy, val_loss, etc)
    # from history dict (contains lists of equal length for each metric over
    # all epoch_results)
    training_csv_header = ["epoch"] + list(history_dict.keys())

    save_dict_to_csv(dict=history_dict, csv_file_path=os.path.join(stats_dir, 'training_history.csv'), fieldnames_header=training_csv_header, start_row_num_from_1=True)

    # with open(os.path.join(stats_dir, 'training_history.csv'), 'w+', newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #
    #     fieldnames_header = ["epoch"] + list(history_dict.keys())
    #     writer.writerow(fieldnames_header)
    #
    #     for epoch in range(len(list(history_dict.values())[0])):
    #         epoch_results = [metric[epoch] for metric in history_dict.values()]
    #         writer.writerow([epoch] + epoch_results)

    # # Create graphs for accuracy and mean average percentage error using matplotlib
    training_results = convert_csv_to_dict(os.path.join(stats_dir, 'training_history.csv'))
    plot(training_results["epoch"], training_results["loss"], xlabel="Epochs", ylabel="Loss", title="Loss vs Epochs", save=True, filename=os.path.join(stats_dir, "loss.png"))

    # graph_acc_and_MAP()
    saved = save_model(model, model_weights_dir)
    if not saved:
        print("didn't save model weights")
    else:
        print("saved model to disk")


if __name__ == '__main__':
    path_to_config='config.yaml'
    config = initialize_hyper(path_to_config)
    print(config)
    if config is None:
        print("error in initialize_hyper")
        sys.exit(1)
    GLOBALS.CONFIG=config

    model, history, results = train()
    # process_outputs(model, history_dict, results, scheduler, dataset, number_of_epochs):
    process_outputs(model=model, history_dict=history.history, results=results, scheduler=config['LR_scheduler'], dataset=config['directory'], number_of_epochs=config['number_of_epochs'])
