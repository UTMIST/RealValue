import cv2
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
import global_vars as GLOBALS
from sklearn.preprocessing import OneHotEncoder

'''
How things were:
One folder --> Create 4-to-1 images out of that, then we'd just create train_images (1-300), val_images(300-400), test_images (400-535)

Data Augmentation on 70% of the Images
Matthew will provide us:
1. Train folder (has augmented)
2. Validation folder
3. Test folder

For each of the above folder 4-to-1, and then appropriately put into train_images, val_images, test_images and other stuff.
'''

# Set individual image sizes here

# IMAGE_HEIGHT = 32
# IMAGE_WIDTH = 32

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
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
    return True

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
    return True
'''
#train_stats, validation_stats, test_stats
def split_stats_data(directory, splitting_parameter):
    df = pd.read_fwf(directory+'/HousesInfo.txt')
    new_df=pd.DataFrame(columns=['Bedrooms','Bathrooms','SqFt','ZipCode','Price'])

    def remove_values_from_list(the_list, val):
       return [value for value in the_list if value != val]

    for index,row in df.iterrows():
        final=row.tolist()[0].split()
        final = list(map(float, final))
        new_df.loc[index]=final

    new_df = new_df.drop('ZipCode', 1)

    print(new_df.head())
    price_min=new_df['Price'].min()
    sqft_min=new_df['SqFt'].min()
    price_max=new_df['Price'].max()
    sqft_max=new_df['SqFt'].max()
    min_max_values={'price_min':price_min,'sqft_min':sqft_min,'price_max':price_max,'sqft_max':sqft_max}

    continuous_feats=['SqFt','Price']
    categorical_feats=['Bedrooms','Bathrooms']
    for i, feature in enumerate(continuous_feats):
        new_df[feature]=(new_df[feature]-new_df[feature].min())/(new_df[feature].max()-new_df[feature].min())
    cts_data=new_df[continuous_feats].values

    for i, feature in enumerate(categorical_feats):
        temp_column=pd.get_dummies(new_df[feature]).to_numpy()
        if i==0:
            cat_onehot=temp_column
        else:
            cat_onehot=np.concatenate((cat_onehot,temp_column),axis=1)
    final_stats_array= np.concatenate([cat_onehot,cts_data], axis=1)
    #final_x_array is following [[one hot vec for beds, one hot vec for bathrooms, sqft as normalized quantity],(...)] #Note that each item in the list is a training example
    #final_price_array is following: [normalized price]
    final_x_array=final_stats_array[:,:-1]
    final_price_array= final_stats_array[:,-1]
    number_of_elements=len(final_x_array)
    num_train = int(number_of_elements * splitting_parameter[0])
    num_validate = int(number_of_elements * splitting_parameter[1])
    num_test = number_of_elements - num_train - num_validate

    train_stats=final_x_array[:num_train]
    validation_stats=final_x_array[num_train:num_train+num_validate]
    test_stats=final_x_array[num_train+num_validate:]

    train_prices=final_price_array[:num_train]
    validation_prices=final_price_array[num_train:num_train+num_validate]
    test_prices=final_price_array[num_train+num_validate:]

    return train_stats,validation_stats,test_stats,train_prices,validation_prices,test_prices,min_max_values
'''
def true_dataframe(directory_path):
    df = pd.read_fwf(directory_path, header=None)
    new_df=pd.DataFrame(columns=['Bedrooms','Bathrooms','SqFt','ZipCode','Price'])

    def remove_values_from_list(the_list, val):
       return [value for value in the_list if value != val]

    for index,row in df.iterrows():
        try:
            final=row.tolist()[0].split()
        except:
            final=(str(row.tolist()[0])+' '+row.tolist()[1]).split()
        final = list(map(float,final))
        '''except:
            final=row.tolist().split()
            print(final)
            print(row.tolist())
            print(row.tolist()[0].split())
            print(final, index)
            a=b'''
        new_df.loc[index]=final
    new_df = new_df.drop('ZipCode', 1)
    return new_df

def split_stats_data(directory, tag = 'train', oneh_encoder = None):
    new_df = true_dataframe(directory+'/'+tag+'_'+'HousesInfo.txt')

    price_min=new_df['Price'].min()
    sqft_min=new_df['SqFt'].min()
    price_max=new_df['Price'].max()
    sqft_max=new_df['SqFt'].max()
    min_max_values={'price_min':price_min,'sqft_min':sqft_min,'price_max':price_max,'sqft_max':sqft_max}

    continuous_feats=['SqFt','Price']
    categorical_feats=['Bedrooms','Bathrooms']
    for i, feature in enumerate(continuous_feats):
        new_df[feature]=(new_df[feature]-new_df[feature].min())/(new_df[feature].max()-new_df[feature].min())
    cts_data=new_df[continuous_feats].values

    '''
    oneh_encoder = OneHotEncoder()
    label_temp=np.array(labels).reshape(-1,1)
    oneh_encoder.fit(np.array(full_loader.dataset.targets).reshape(-1,1))
    labels=torch.from_numpy(oneh_encoder.transform(label_temp).toarray()).float()

    data_temp=new_df[categorical_feats]
    oneh_encoder.fit(data_temp)
    cat_onehot1=oneh_encoder.transform(data_temp).toarray()
    print(cat_onehot1)
    '''

    cat_onehot=oneh_encoder.transform(new_df[categorical_feats]).toarray()
    '''
    for i, feature in enumerate(categorical_feats):
        temp_column=pd.get_dummies(new_df[feature]).to_numpy()
        if i==0:
            cat_onehot=temp_column
        else:
            cat_onehot=np.concatenate((cat_onehot,temp_column),axis=1)
    '''
    final_stats_array= np.concatenate([cat_onehot,cts_data], axis=1)
    #final_x_array is following [[one hot vec for beds, one hot vec for bathrooms, sqft as normalized quantity],(...)] #Note that each item in the list is a training example
    #final_price_array is following: [normalized price]
    final_x_array=final_stats_array[:,:-1]
    final_price_array= final_stats_array[:,-1]

    return final_x_array, final_price_array, min_max_values, oneh_encoder

#just pass in main_dataset/final as directory I think
#splitting_paramter [train_ratio, val_ratio, test_ratio] example [0.7, 0.15, 0.15]
def split_image_data(directory, tag='train'):
    file_list = os.listdir(directory)
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
    images = []
    # first_img_file = file_list[0]
    # first_img_file_path = os.path.join(directory, first_img_file)
    # first_img = cv2.imread(first_img_file_path)
    # train_images = np.copy(first_img)
    # #np.zeros(first_img.shape, first_img.dtype)
    # validation_images = np.copy(first_img)
    # test_images = np.copy(first_img)

    # print(file_list)
    # print(directory)

    # print(first_img_file)
    # print(first_img)
    # print(first_img.shape)
    # print(first_img.dtype)

    #iterate through all images in the folder
    img_counter = 0
    for img_file in file_list:
        img_file_path = os.path.join(directory, img_file)
        img = cv2.imread(img_file_path)
        img_counter += 1
        images.append(img)

    images = np.array(images)
    # visualize different images in split dataset to ensure proper shape
    #display_image(images[20])
    return images

def return_splits(directories):
    [train_directory, val_directory, test_directory]=directories
    train_directory_final = train_directory + '_final'
    val_directory_final = val_directory + '_final'
    test_directory_final = test_directory + '_final'
    #Compile 4 images into one and save them in respective directory (e.g 'train_directory_final' for train images)
    compile_full_image(train_directory,train_directory_final)
    compile_full_image(val_directory,val_directory_final)
    compile_full_image(test_directory,test_directory_final)
    #Fetch images from train,valid,test images
    train_images = split_image_data(train_directory_final,"Train")
    validation_images = split_image_data(val_directory_final,"Validation")
    test_images = split_image_data(test_directory_final,"Test")
    #Locate path of full houseinfo.txt
    dataset_name = 'raw_dataset'
    house_info = 'HousesInfo.txt'
    current_working_dir = os.getcwd() #current working directory
    dataset_full_path = os.path.join(current_working_dir, dataset_name) #FULL path of the original dataset
    house_info_path = os.path.join(dataset_full_path,house_info)
    #Initialize OneHotEncoder and fit to full houseinfo.txt for categorical features
    df = true_dataframe(house_info_path)
    categorical_feats=['Bedrooms','Bathrooms']
    oneh_encoder = OneHotEncoder()
    oneh_encoder.fit(df[categorical_feats])
    #Fetch stats from train,valid,test images
    train_stats, train_prices, train_min_max, train_oneh_encoder = split_stats_data(train_directory, tag='train',oneh_encoder = oneh_encoder)
    validation_stats, validation_prices, validation_min_max, train_oneh_encoder = split_stats_data(val_directory, tag='val',oneh_encoder = oneh_encoder)
    test_stats, test_prices, test_min_max, train_oneh_encoder = split_stats_data(test_directory, tag='test',oneh_encoder = oneh_encoder)
    #Create dict with all required information
    main_dict={'train_images':train_images,'train_stats':train_stats,'train_prices':train_prices,'validation_images':validation_images,'validation_stats':validation_stats,'validation_prices':validation_prices,'test_images':test_images,'test_images':test_images,'test_stats':test_stats,'test_prices':test_prices, 'train_min_max':train_min_max,'validation_min_max':validation_min_max,'test_min_max':test_min_max}
    return main_dict

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
    '''compile_full_image("raw_dataset", "raw_dataset/final")'''
