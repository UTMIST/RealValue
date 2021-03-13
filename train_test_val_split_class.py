import numpy as np
import math
import os
import shutil

class train_test_val_split_class:
    def __init__(self, dataset_dir_name, txt_filename, ratio):
        """
        INPUTS:
            ::string:: dataset_dir_name         #FULL path of the original dataset
            ::string:: txt_filename             #Base path (not full path) of the txt label file
            ::tuple of 3 floats:: ratio         #tuple of 3 elements containing the train, val, test ratio
        """
        self.dataset_dir_name = dataset_dir_name
        self.root = os.path.dirname(self.dataset_dir_name) #directory in which dataset_dir_name is in
        self.dataset_dir = os.path.join(self.root, dataset_dir_name) #directory of original dataset

        # self.splitted_dataset_dir = os.path.join(self.root, 'splitted_dataset') #directory of splitted dataset

        #OLD (2020):
        #self.splitted_dataset_dir = os.path.join(self.root, 'splitted_dataset_' + str(ratio[0]) + '_' + str(ratio[1]) + '_' + str(ratio[2])) #directory of splitted dataset
        #NEW (Jan. 31, 2021):
        self.splitted_dataset_dir = os.path.join(self.root, 'splitted_' + os.path.basename(dataset_dir_name) + '_' + str(ratio[0]) + '_' + str(ratio[1]) + '_' + str(ratio[2])) #directory of splitted dataset

        self.train_dir = os.path.join(self.splitted_dataset_dir, 'train')
        self.val_dir = os.path.join(self.splitted_dataset_dir, 'val')
        self.test_dir = os.path.join(self.splitted_dataset_dir, 'test')
        self.txt_filename = os.path.join(self.dataset_dir, txt_filename) #full filename of txt label file
        self.ratio = ratio #tuple of 3 elements containing: train, val, test ratio
        np.random.seed(1000)

    def get_split_houses(self, ratio):
        """
        Returns 3 lists which are the house numbers for train, val, test
        INPUTS:
            ::tuple of size 3:: ratio           #train, val, test
        OUTPUTS:
            ::list:: train_house_numbers        #train
            ::list:: val_house_numbers          #val
            ::list:: house_numbers              #test
        """
        num_images = 0 #total number of images
        for entry in os.scandir(self.dataset_dir):
            if entry.is_file() and (not entry.path.endswith(".txt")):
                num_images += 1
        num_houses = int(num_images / 4) #total number of houses
        print(num_houses)

        num_train = int(math.floor(num_houses * ratio[0])) #number of houses for training
        num_val = int(math.ceil(num_houses * ratio[1])) #number of houses for validation
        num_test = int(math.floor(num_houses * ratio[2])) #number of houses for test

        print("train, val, test:", num_train, num_val, num_test, "\n\n")

        house_numbers = [i for i in range(1,num_houses+1,1)] #list of numbers from 1 to num_houses

        train_house_numbers = [] #house numbers for train
        for i in range(0,num_train,1): #randomly pick num_train houses for train
            pick_num = np.random.randint(0,len(house_numbers))
            pick = house_numbers[pick_num]
            train_house_numbers += [pick]
            house_numbers.remove(pick)

        val_house_numbers = [] #house numbers for val
        for i in range(0,num_val,1): #randomly pick num_val houses for val
            pick_num = np.random.randint(0,len(house_numbers))
            pick = house_numbers[pick_num]
            val_house_numbers += [pick]
            house_numbers.remove(pick)

        #Note: whatever house numbers that have not been picked will be for test
        '''
        print(train_house_numbers, "\n")
        print(val_house_numbers, "\n")
        print(house_numbers, "\n")
        print("Total:", len(train_house_numbers) + len(val_house_numbers) + len(house_numbers))
        '''
        return train_house_numbers, val_house_numbers, house_numbers #train, val, test

    def write_to_test(self, test_list, house_info_filename, result_filename):
        house_info = self.get_house_info_list(house_info_filename)
        f = open(result_filename, 'w')
        for house_num in test_list:
            target_house_info = house_info[house_num-1]
            f.write(target_house_info)
        f.close()
        return True

    def get_house_info_list(self, filename):
        #Reading in file
        f = open(filename, "r")
        housing_info_list = []
        for line in f:
            housing_info_list.append(line)
        f.close()
        return housing_info_list

    #get_house_info_list(txt_filename)
    def train_test_val_split(self, ratio, house_info_filename):
        if os.path.isdir(self.splitted_dataset_dir) == False:
            os.mkdir(self.splitted_dataset_dir)
        if os.path.isdir(self.train_dir) == False:
            os.mkdir(self.train_dir)
        if os.path.isdir(self.test_dir) == False:
            os.mkdir(self.test_dir)
        if os.path.isdir(self.val_dir) == False:
            os.mkdir(self.val_dir)

        train_house_numbers, val_house_numbers, test_house_numbers = self.get_split_houses(ratio)
        self.write_to_test(train_house_numbers, house_info_filename, os.path.join(self.train_dir, 'train_HousesInfo.txt'))
        self.write_to_test(val_house_numbers, house_info_filename, os.path.join(self.val_dir, 'val_HousesInfo.txt'))
        self.write_to_test(test_house_numbers, house_info_filename, os.path.join(self.test_dir, 'test_HousesInfo.txt'))

        for entry in os.scandir(self.dataset_dir):
            if entry.is_file() and (not entry.path.endswith(".txt")):
                entry_string = os.path.basename(entry) #just the filename, not the full path
                splitted = entry_string.split("_")
                if splitted[0]=='.DS':
                    continue
                else:
                    filenum = int(splitted[0]) #file number of current file

                if filenum in train_house_numbers:
                    shutil.copy(entry.path,self.train_dir) #copy image to train_dir
                elif filenum in val_house_numbers:
                    shutil.copy(entry.path,self.val_dir) #copy image to val_dir
                elif filenum in test_house_numbers:
                    shutil.copy(entry.path,self.test_dir) #copy image to test_dir

        return True

    def do_split(self):
        self.train_test_val_split(self.ratio, self.txt_filename)
        return True

#if you only want to do the split and no augmentation, uncomment below:
'''
dataset_full_path = 'C:/Users/Matthew/Desktop/UTMIST/raw_dataset' #change the path accordingly
train_val_test_ratio = (0.70,0.10,0.20)
txt_filename_raw = 'HousesInfo.txt'
obj = train_test_val_split_class(dataset_full_path, txt_filename_raw, train_val_test_ratio)
obj.do_split()
'''
