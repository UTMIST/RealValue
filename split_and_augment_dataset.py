import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import global_vars as GLOBALS

from rotate_and_crop import rotate_and_crop as rotcrop
from train_test_val_split_class import train_test_val_split_class as train_test_val_split_class
# from pipeline import initialize_hyper

def gen_augmented_dataset(i, dataset_dir_name, augmented_dir_name, seed_num):
    """
    Function creates 1 set of augmented images from images in dataset_dir_name
    INPUTS:
        ::int:: i                       #the augmented dataset number
        ::string:: dataset_dir_name     #FULL path of the original dataset
        ::string:: augmented_dir_name   #name of directory of augmented dataset, which will be created in parent folder of dataset_dir_name
        ::int:: seed_num                #random seed number
    """
    np.random.seed(seed_num) #Change the random seed to get a new set of images

    root = os.path.dirname(dataset_dir_name) #directory in which dataset_dir_name is in
    dataset_dir = dataset_dir_name #directory of original dataset
    augmented_dir = os.path.join(root, augmented_dir_name) #directory of augmented images
    if os.path.isdir(augmented_dir) == False: #if directory does not exist, make director
        os.mkdir(augmented_dir)

    last_img_num = 535 * i

    start_time = time.time()
    for entry in os.scandir(dataset_dir):
        if entry.is_file() and (not entry.path.endswith(".txt")):
            entry_string = os.path.basename(entry)
            image = plt.imread(entry.path)

            splitted = entry_string.split("_")
            new_filenum = int(splitted[0]) + last_img_num #new file number

            ####################################################################
            #Data augmentation:
            angle = np.random.uniform(-15,15) #angle to rotate and crop (degrees)
            new_image = rotcrop(image,angle)
            if np.random.randint(2) == 1:
                new_image = tf.image.flip_left_right(new_image) #randomly flip the image along vertical axis
            brightnes_amount = np.random.uniform(0,0.2) #brightness amount, float between [0,1)
            new_image = tf.image.adjust_brightness(new_image, brightnes_amount) #change brightness
            saturation_amount = np.random.uniform(1,2) #saturation amount
            new_image = tf.image.adjust_saturation(new_image, saturation_amount) #change saturation
            ####################################################################

            tf.keras.preprocessing.image.save_img(os.path.join(augmented_dir, str(new_filenum) + "_" + splitted[1]), new_image) #save image
    end_time = time.time()
    print("Augmented dataset number " + str(i) + ": Augmenting all the images took", end_time-start_time, "seconds")
    return True

def augment_dataset(n, dataset_dir_name):
    """
    Function creates n sets of augmented images from images in dataset_dir_name
    INPUTS:
        ::int:: n                       #number of sets of augmented images to create
        ::string:: dataset_dir_name     #FULL path of the original dataset
    """
    for i in range(1,n+1,1):
        seed_num = i*10
        gen_augmented_dataset(i, dataset_dir_name, "augmented" + str(i).zfill(2), seed_num)

    return True

def merge_augmented_datasets(n, dataset_dir_name, merged_dir_name, txt_filename):
    """
    This function merges the augmented datasets into one folder
    INPUTS:
        ::int:: n                           #number of sets of augmented images to create
        ::string:: dataset_dir_name         #FULL path of the original dataset
        ::string:: merged_dir_name          #Name of the merged directory of augmented images
        ::string:: txt_filename             #Base path (not full path) of the txt label file
    """

    root = os.path.dirname(dataset_dir_name) #parent directory in which dataset_dir_name is in
    ma_dir = os.path.join(root, merged_dir_name) #merged directory of augmented images
    if os.path.isdir(ma_dir) == False: #if directory does not exist, make director
        os.mkdir(ma_dir)
    full_txt_filename = os.path.join(dataset_dir_name, txt_filename) #full filename of txt label file
    merged_txt_filename = os.path.join(ma_dir, txt_filename) #path of txt file containing all the merged labels

    #copy files from dataset_dir_name to to ma_dir
    for base_filename in os.listdir(dataset_dir_name):
        if base_filename[-3:len(base_filename)] != 'txt':
            full_filename = os.path.join(dataset_dir_name, base_filename)
            if os.path.isfile(full_filename):
                shutil.copy(full_filename, ma_dir)

    #copy files from augmented dataset to ma_dir
    for i in range(1,n+1,1):
        augmented_dir = os.path.join(root, "augmented" + str(i).zfill(2))
        for base_filename in os.listdir(augmented_dir):
            if base_filename[-3:len(base_filename)] != 'txt':
                full_filename = os.path.join(augmented_dir, base_filename)
                if os.path.isfile(full_filename):
                    shutil.copy(full_filename, ma_dir)

    house_info = get_house_info_as_list(full_txt_filename)
    #copy contents of the original txt label file
    for i in range(0,n+1,1):
        with open(merged_txt_filename, 'a') as f:
            for line in house_info:
                f.write(line)
    return True

def get_house_info_as_list(filename):
    """
    INPUT:
        ::string:: filename         #FULL filename of txt file
    OUTPUT:
        ::list:: housing_info_list  #list containing the lines of filename
    """
    housing_info_list = []
    with open(filename, 'r') as f:
        for line in f:
            housing_info_list.append(line)
    return housing_info_list

def split_and_augment_train_dataset(ratio, dataset_full_path, txt_filename_raw, n, split=True, augment=True):
    """
    This function splits the original dataset and then applies data augmentation
    to the train dataset. The final dataset containing augmented train images
    and labels is called train_augmented
    INPUTS:
        ::tuple of 3 floats:: ratio         #train, val, test split ratio
        ::string:: dataset_full_path        #FULL path of the original dataset
        ::string:: txt_filename_raw         #Base path (not full path) of the txt label file
        ::int:: n                           #number of sets of augmented images to create
        ::boolean:: split                   #Whether or not to split the dataset
        ::boolean:: augment                 #Whether or not to augment the train dataset
    """
    train_dir = ''
    if split == True:
        start_time = time.time()
        # splitter = train_test_val_split_class(dataset_full_path, txt_filename_raw, train_val_test_ratio)
        splitter = train_test_val_split_class(dataset_full_path, txt_filename_raw, ratio)
        splitter.do_split()
        end_time = time.time()
        print("Splitting the dataset took " + str(end_time-start_time) + " seconds")
        train_dir = splitter.train_dir

    if train_dir == '':
        train_dir =  os.path.join(os.path.dirname(dataset_full_path), 'splitted_dataset', 'train')
        if os.path.isdir(train_dir) == False:
            print("ERROR: train_dir does not exist")
            return False

    if augment == True:
        augment_dataset(n, train_dir)

    start_time = time.time()
    merge_augmented_datasets(n, train_dir, 'train_augmented', 'train_' + txt_filename_raw)
    end_time = time.time()
    print('Merging the augmented datasets took ' + str(end_time-start_time) + ' seconds')
    return True

################################################################################

'''
if __name__ == '__main__':
    #CHANGE THIS STUFF IF NEEDED:
    config = initialize_hyper('config.yaml')
    print(config)
    if config is None:
        print("error in initialize_hyper")
        sys.exit(1)
    GLOBALS.CONFIG=config

    # n = 2 #number of times to augment the original train set
    n = GLOBALS.CONFIG['augmentation_multiplier'] - 1
    dataset_name = 'raw_dataset' #name of the dataset
    train_val_test_ratio = GLOBALS.CONFIG['train_val_test_ratio']#(0.70,0.10,0.20) #train, val, test ratio
    txt_filename_raw = 'HousesInfo.txt' #name of the txt label file in the original dataset

    ############################################################################

    #It is assumed that this script is in the same directory as the raw_dataset
    current_working_dir = os.getcwd() #current working directory
    dataset_full_path = os.path.join(current_working_dir, dataset_name) #FULL path of the original dataset
    split_and_augment_train_dataset(train_val_test_ratio, dataset_full_path, txt_filename_raw, n, split=True, augment=True)
'''
