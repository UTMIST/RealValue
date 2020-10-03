import cv2
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Set individual image sizes here

# IMAGE_HEIGHT = 32
# IMAGE_WIDTH = 32

IMAGE_HEIGHT = 120
IMAGE_WIDTH = 120
#
# IMAGE_HEIGHT = 720
# IMAGE_WIDTH = 720

MERGED_IMAGE_HEIGHT = 2 * IMAGE_HEIGHT
MERGED_IMAGE_WIDTH = 2 * IMAGE_WIDTH

def loadImages(folder):
    images = []
    file_names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            file_names.append(filename)
    return file_names,images

def resize_image(img, method="squeeze"):
    # either crop & scale or squeeze input image to IMAGE_HEIGHT x IMAGE_WIDTH

    height, width, channels = img.shape

    if method == "squeeze":
        res = cv2.resize(img,(IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
    elif method == "crop":
         res = img[0:IMAGE_HEIGHT, 0: IMAGE_WIDTH]
    else:
        res = img


    return res

def merge_part_images(raw_img_dict):
    output_dict = {}

     # img11 | img12
     # ------------------
     # img21 | img22

     # bathroom | frontal
     # ------------------
     # bedroom | kitchen

     # currently have
     # img[0] | img[2]
     # ------------------
     # img[1] | img[3]

     #100: [img1,img2,img3,img4]
     #100: img_full

    for file_name, img_list in raw_img_dict.items():
        merged_image = np.zeros((MERGED_IMAGE_HEIGHT, MERGED_IMAGE_WIDTH, 3), np.uint8)#img_list[0].dtype)

        merged_image[0:IMAGE_HEIGHT, 0: IMAGE_WIDTH] = img_list[0]
        merged_image[IMAGE_HEIGHT: MERGED_IMAGE_HEIGHT, 0: IMAGE_WIDTH] = img_list[1]
        merged_image[0:IMAGE_HEIGHT, IMAGE_WIDTH: MERGED_IMAGE_WIDTH] = img_list[2]
        merged_image[IMAGE_HEIGHT: MERGED_IMAGE_HEIGHT, IMAGE_WIDTH: MERGED_IMAGE_WIDTH] = img_list[3]

        output_dict[file_name] = merged_image

    return output_dict

def display_image(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)

def write_files_to_folder(merged_image_dict, output_dir):
    if not os.path.exists(os.path.join(os.path.dirname(__file__), output_dir)):
        os.makedirs(output_dir)

    for file_num, merged_image in merged_image_dict.items():
        filename = "%d.png" % file_num
        file_path = os.path.join(output_dir, filename)
        cv2.imwrite(file_path, merged_image)

def compile_full_image(folder_path, output_dir):
    file_names,images=loadImages(folder_path)
    resized_images_dict=defaultdict(list)
    for i in range(0,len(images)):
        resized_image=resize_image(images[i])
        file_name=file_names[i]
        resized_images_dict[int(file_name[0:file_name.index("_")])].append(resized_image)
    #print(resized_images_dict.keys())
    merged_images=merge_part_images(resized_images_dict)
    write_files_to_folder(merged_images, output_dir)

def split_stats_data():
    pass
def split_price_data():
    pass

#just pass in main_dataset/final as directory I think
#splitting_paramter [train_ratio, val_ratio, test_ratio] example [0.7, 0.15, 0.15]
def split_image_data(directory, splitting_parameter):
    file_list = os.listdir(directory)
    num_train = int(len(file_list) * splitting_parameter[0])
    num_validate = int(len(file_list) * splitting_parameter[1])
    num_test = len(file_list) - num_train - num_validate
    '''
    get the dimensions of the image from reading 1 image in the folder
    for image_name in file_list:
        try:
            img_shape = cv2.imread(image_name).shape
        except:
            print("error reading a single image")
        break
    (rows, cols, channels) = img_shape
    train_images = np.zeros(shape=(num_train, rows, cols, channels))
    validation_images = np.zeros(shape=(num_validate, rows, cols, channels))
    test_images = np.zeros(shape=(num_test, rows, cols, channels))
    '''
    train_images = []
    test_images = []
    validation_images = []
    #iterate through all images in the folder
    img_counter = 0
    for img_file in file_list:
        img = cv2.imread(image_file)
        img_counter += 1
        if len(train_images) < num_train:
            train_images.append(img)
            continue
        if len(validation_images) < num_validate:
            validation_images.append(img)
            continue
        if len(test_images) < num_test:
            test_images.append(img)
            continue
    if len(train_images)==num_train and len(validation_images)==num_validate and len(test_images)==num_test:
        print("correctly split")
    else:
        print("splitting was incorrect")

    train_images = np.array(train_images)
    validation_images = np.array(validation_images)
    test_images = np.array(test_images)

    print("train images shape:", train_images.shape)
    print("validation images shape:", validation_images.shape)
    print("test images shape:", test_images.shape)

    return train_images, validation_images, test_images

def return_splits():
  '''
  Inputs: main_dataset_path, train/test and validation splits.
  Outputs: train_images, train_stats, train_prices, validation_images, validation_stats, validation_prices, test_images, test_stats, test_prices.

  The goal of this function is to return all inputs and outputs required for training.
  The process occurs in the following steps:
  1. For the dataset in main_dataset_path, convert the 4 images corresponding to 1 house into 1 image per house
    - Bedroom top left, bathroom top right, kitchen bottom left, frontal top right.
    - Write these images to main_dataset_path/final
  2. Use the "main_dataset/final" folder, the train_test splits and main_dataset/HousesInfo.txt to split the dataset appropriately into training, validation and test sections
    -train_images is a 4D numpy array
        -1st dimension is number of loadImages
        -2nd and 3rd dims are height and width (535 * 0.7, 64, 64, 3)
        -4th dim is channels
    -train_stats is a (535 * 0.7, 3)
    -train_prices (535 * 0.7, 1)
    -normalize using standard normalization formula
        -for train_stats, one hot encode the number of bedrooms, Bathrooms, normalize squarefootage
        -ex. 4 different samples, with 0, 1, 2, 3 bedrooms


  3. Return all of the above required Outputs within a Dictionary.
    - aka {"train_images":train_images, "train_stats":train_stats}
  '''


if __name__ == "__main__":

    #compile_full_image("toronto_dataset", "processed_dataset")
    compile_full_image("raw_dataset", "processed_dataset")
