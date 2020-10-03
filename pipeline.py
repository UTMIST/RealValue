import tensorflow as tf
from keras.models import Model
import time
import yaml
import csv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# from models import get_network
from models.CNN_models.lenet import LeNet
from models.dense_models.simple_densenet import SimpleDenseNet


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

def train(path_to_config):
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
    if config is None:
        print("error in initialize_hyper")
        sys.exit(1)

    train_images, train_stats, train_prices, validation_images, validation_stats, validation_prices, \
        test_images, test_stats, test_prices = initialize_datasets( ... )

    CNN_type = config['network']
    CNN = get_network(CNN_type)
    Dense_NN = SimpleDenseNet(train_stats.shape[1])
    Multi_Input = concatenate([Dense_NN.output, CNN.output])

    Final_Fully_Connected_Network = Dense(4, activation = 'relu')(Multi_Input)
    Final_Fully_Connected_Network = Dense(1)(Final_Fully_Connected_Network)

    model = Model(inputs = [Dense_NN.input , CNN.input], outputs = Final_Fully_Connected_Network)
    model.compile(optimizer = config['optimizer'], loss = config['loss'],
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.MeanAbsolutePercentageError()])
    history = model.fit([train_stats,train_images], train_prices, validation_split = config['validation_split'],
            epochs = config['epochs'],
            batch_size = config['batch_size'],
            callbacks= [tensorboard]) #not sure if we have the tensorfboard callback

    #I'm not sure how we are incorporating the validation dataset into our training code?
    preds = model.predict([test_stats,test_images])



    # results = model.evaluate (with hyperparameters & test results from 2)
    # btw, I also added these lines below for some other metrics I need for plotting
    # i guess we can ask Arsh where the model evaluation code should be ok
    result = model.evaluate(test_dataset)
    evaluation_results = dict(zip(model.metrics_names, result))

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

def process_outputs(model, history_dict, results, model_name, learning_rate, scheduler, dataset, number_of_epochs):
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

    #why do we need to convert the model back to YAML? it seems like we just want a model_weights.h5
    # Save model
    #
    # # serialize model to YAML
    # model_yaml = model.to_yaml()
    # with open(Model_Name_yml, "w+") as yaml_file:
    #     yaml_file.write(model_yaml)
    #
    # # serialize weights to HDF5
    # model.save_weights("model_weights.h5")
    # print("Saved model to disk")
    #
    #
    # save_training_validation_test_results()


if __name__ == '__main__':
    process_outputs(model="blah", history_dict={'loss': [0.3379512131214142, 0.16062788665294647],
 'sparse_categorical_accuracy': [0.9043999910354614, 0.9524800181388855],
 'val_loss': [0.18728218972682953, 0.17670848965644836],
 'val_sparse_categorical_accuracy': [0.9452000260353088, 0.9470999836921692]}, results="blah", model_name="test", learning_rate=0.05, scheduler="Adam", dataset="MNIST", number_of_epochs=500)
    # Model_Name = "Final_House_price_estimation- {}".format(int(time.time()))
    # Model_Name_yml = "Final_House_price_estimation- {}.yaml".format(int(time.time()))
    # #tensorboard = TensorBoard(log_dir= 'logs/{}'.format(Model_Name))
    #
    # #X_train , X_test , y_train , y_test , train_images, test_images, y_test_actual = Prepare_Final_Data("Houses Dataset/HousesInfo.txt")
    # Dense_NN = SimpleDenseNet(X_train.shape[1])
    # CNN = LeNet()
    #
    # Multi_Input = concatenate([Dense_NN.output, CNN.output])
    # Final_Fully_Connected_Network = Dense(4, activation = 'relu')(Multi_Input)
    # Final_Fully_Connected_Network = Dense(1)(Final_Fully_Connected_Network)
    # model = Model(inputs = [Dense_NN.input , CNN.input], outputs = Final_Fully_Connected_Network)
    # model.compile(optimizer = 'adam', loss = 'mse')
    # model.fit([X_train,train_images], y_train, validation_split = 0.15,
    #         epochs = 50,
    #         batch_size = 8,
    #         callbacks= [tensorboard])
    #
    #
    #
    # preds = model.predict([X_test,test_images])
    # error = preds.flatten() - y_test
    # squared_error = error ** 2
    # MSE = np.mean(squared_error)
    #
    # #train(Model_Name_yml)
    #
    # r2_score_test = r2_score(y_test,preds.flatten())
    #
    #
    # # compute the difference between the *predicted* house prices and the
    # # *actual* house prices, then compute the percentage difference and
    # # the absolute percentage difference
    # diff = preds.flatten() - y_test
    # percentDiff = (diff / y_test) * 100
    # absPercentDiff = np.abs(percentDiff)
    #
    # # compute the mean and standard deviation of the absolute percentage
    # # difference
    # mean = np.mean(absPercentDiff)
    # std = np.std(absPercentDiff)

    # # finally, show some statistics on our model
    # locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    # print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
    # print("MSE = ", MSE)
    # print("R2_score = ", r2_score_test)
