import numpy as np
import math
import os
import shutil

np.random.seed(1000)

dataset_dir_name = 'raw_dataset'
root = os.getcwd() #current working directory
dataset_dir = os.path.join(root, dataset_dir_name) #directory of original dataset
splitted_dataset_dir = os.path.join(root, 'splitted_dataset')
train_dir = os.path.join(splitted_dataset_dir, 'train')
val_dir = os.path.join(splitted_dataset_dir, 'val')
test_dir = os.path.join(splitted_dataset_dir, 'test')
txt_filename = os.path.join(dataset_dir, 'HousesInfo.txt')
ratio = (0.775, 0.075, 0.15) #train, val, test


def get_split_houses(ratio):
    """
    INPUTS:
        ::tuple of size 3:: ratio           #train, val, test
    OUTPUTS:
        ::list:: train_house_numbers        #train
        ::list:: val_house_numbers          #val
        ::list:: house_numbers              #test
    """
    num_images = 0 #total number of images
    for entry in os.scandir(dataset_dir):
        if entry.is_file() and (not entry.path.endswith(".txt")):
            num_images += 1
    num_houses = int(num_images / 4) #total number of houses

    num_train = int(math.floor(num_houses * ratio[0])) #number of houses for training
    num_val = int(math.ceil(num_houses * ratio[1])) #number of houses for validation
    num_test = int(math.floor(num_houses * ratio[2])) #number of houses for test

    print("train, val, test:", num_train, num_val, num_test, "\n\n")

    house_numbers = [i for i in range(1,num_houses+1,1)] #list of numbers from 1 to num_houses

    train_house_numbers = []
    for i in range(0,num_train,1):
        pick_num = np.random.randint(0,len(house_numbers))
        pick = house_numbers[pick_num]
        train_house_numbers += [pick]
        house_numbers.remove(pick)

    val_house_numbers = []
    for i in range(0,num_val,1):
        pick_num = np.random.randint(0,len(house_numbers))
        pick = house_numbers[pick_num]
        val_house_numbers += [pick]
        house_numbers.remove(pick)
    '''
    print(train_house_numbers, "\n")
    print(val_house_numbers, "\n")
    print(house_numbers, "\n")
    print("Total:", len(train_house_numbers) + len(val_house_numbers) + len(house_numbers))
    '''
    return train_house_numbers, val_house_numbers, house_numbers #train, val, test

def write_to_test(test_list, house_info_filename, result_filename):
    house_info = get_house_info_list(house_info_filename)
    f = open(result_filename, 'w')
    for house_num in test_list:
        target_house_info = house_info[house_num-1]
        f.write(target_house_info)
    f.close()
    return True

def get_house_info_list(filename):

    #Reading in file
    f = open(filename, "r")
    housing_info_list = []
    for line in f:
        housing_info_list.append(line)
    f.close()
    return housing_info_list

#get_house_info_list(txt_filename)
def train_test_val_split(ratio, house_info_filename):
    global splitted_dataset_dir
    global train_dir
    global test_dir
    global val_dir
    if os.path.isdir(splitted_dataset_dir) == False:
        os.mkdir(splitted_dataset_dir)
    if os.path.isdir(train_dir) == False:
        os.mkdir(train_dir)
    if os.path.isdir(test_dir) == False:
        os.mkdir(test_dir)
    if os.path.isdir(val_dir) == False:
        os.mkdir(val_dir)

    train_house_numbers, val_house_numbers, test_house_numbers = get_split_houses(ratio)
    write_to_test(train_house_numbers, house_info_filename, os.path.join(train_dir, 'train_HousesInfo.txt'))
    write_to_test(val_house_numbers, house_info_filename, os.path.join(val_dir, 'val_HousesInfo.txt'))
    write_to_test(test_house_numbers, house_info_filename, os.path.join(test_dir, 'test_HousesInfo.txt'))

    for entry in os.scandir(dataset_dir):
        if entry.is_file() and (not entry.path.endswith(".txt")):
            entry_string = os.path.basename(entry) #just the filename, not the full path
            splitted = entry_string.split("_")
            filenum = int(splitted[0]) #file number of current file

            if filenum in train_house_numbers:
                shutil.copy(entry.path,train_dir) #copy image to train_dir
            elif filenum in val_house_numbers:
                shutil.copy(entry.path,val_dir) #copy image to val_dir
            elif filenum in test_house_numbers:
                shutil.copy(entry.path,test_dir) #copy image to test_dir

    return True

train_test_val_split(ratio, txt_filename)
