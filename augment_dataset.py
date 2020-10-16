import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from rotate_and_crop import rotate_and_crop as rotcrop

def gen_augmented_dataset(i, dataset_dir_name, augmented_dir_name, seed_num):
    """
    Function creates 1 set of augmented images from images in dataset_dir_name
    INPUTS:
        ::int:: i                       #the augmented dataset number
        ::string:: dataset_dir_name     #name of directory of dataset, relative to cwd
        ::string:: augmented_dir_name   #name of directory of augmented dataset, relative to cwd
        ::int:: seed_num                #random seed number
    """
    np.random.seed(seed_num) #Change the random seed to get a new set of images

    root = os.getcwd() #current working directory
    dataset_dir = os.path.join(root, dataset_dir_name) #directory of original dataset
    augmented_dir = os.path.join(root, augmented_dir_name) #directory of augmented images
    if os.path.isdir(augmented_dir) == False: #if directory of
        os.mkdir(augmented_dir)

    last_img_num = 535 * (i-1)

    start_time = time.time()
    for entry in os.scandir(dataset_dir):
        if entry.is_file() and (not entry.path.endswith(".txt")):
            entry_string = os.path.basename(entry)
            image = plt.imread(entry.path)

            splitted = entry_string.split("_")
            new_filenum = int(splitted[0]) + last_img_num #new file number

            angle = np.random.uniform(-15,15) #angle to rotate and crop (degrees)
            new_image = rotcrop(image,angle)
            if np.random.randint(2) == 1:
                new_image = tf.image.flip_left_right(new_image) #randomly flip the image along vertical axis
            brightnes_amount = np.random.uniform(0,0.2) #brightness amount, float between [0,1)
            new_image = tf.image.adjust_brightness(new_image, brightnes_amount) #change brightness
            saturation_amount = np.random.uniform(1,2) #saturation amount
            new_image = tf.image.adjust_saturation(new_image, saturation_amount) #change saturation

            tf.keras.preprocessing.image.save_img(os.path.join(augmented_dir, str(new_filenum) + "_" + splitted[1]), new_image) #save image
    end_time = time.time()
    print("Augmented dataset number " + str(i) + ": Augmenting all the images took", end_time-start_time, "seconds")
    return True

def augment_dataset(n, dataset_dir_name):
    """
    Function creates n sets of augmented images from images in dataset_dir_name
    INPUTS:
        ::int:: n                       #number of sets of augmented images to create
        ::string:: dataset_dir_name     #name of directory of dataset, relative to cwd
    """
    for i in range(1,n+1,1):
        seed_num = i*10
        gen_augmented_dataset(i, dataset_dir_name, "augmented" + str(i).zfill(2), seed_num)
    return True

if __name__ == "__main__":
    augment_dataset(2, 'raw_dataset')
    '''
    8 sets of collages (for step 2)

    Create 8 different folders
    In each folder there are augmentated images in which ALL transformations are randomly applied
    like ImageDataGenerator. Use a different random seed each time to make a new folder

    Combine images from each folder into ONE final folder (change the naming/numbering of the files)
    Modfiy the txt file accordingly to get ONE txt file
    The end result is ONE folder of 2140*8=17120 images plus ONE txt file
    '''
