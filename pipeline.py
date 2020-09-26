import tensorflow as tf
from keras.models import Model
import time
import yaml
import csv
import sys
import os
import matplotlib.pyplot as plt

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
    with open(os.path.join(stats_dir, 'training_history.csv'), 'w+', newline='') as csv_file:
        writer = csv.writer(csv_file)

        fieldnames_header = history_dict.keys()
        writer.writerow(fieldnames_header)

        for epoch in range(len(list(history_dict.values())[0])):
            epoch_results = [metric[epoch] for metric in history_dict.values()]
            writer.writerow(epoch_results)

if __name__ == '__main__':
    process_outputs(model="blah", history_dict={'loss': [0.3379512131214142, 0.16062788665294647],
 'sparse_categorical_accuracy': [0.9043999910354614, 0.9524800181388855],
 'val_loss': [0.18728218972682953, 0.17670848965644836],
 'val_sparse_categorical_accuracy': [0.9452000260353088, 0.9470999836921692]}, results="blah", model_name="test", learning_rate=0.05, scheduler="Adam", dataset="MNIST", number_of_epochs=500)
