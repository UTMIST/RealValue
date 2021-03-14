# RealValue
## About

RealValue is a machine learning project for predicting home prices in Toronto. Using TensorFlow convolutional neural networks in conjunction with a dense network component, owners can take a couple of pictures of their home, enter a few simple details and they will be provided an accurate price range of what their home is worth. This ease of use allows homeowners to be confident about their residential decisions and be more informed about the real estate market than ever before.

For more details about our project, please take a look at our Medium article <insert link here> and check out our website at myrealvalue.ca
## Motivation and Goal
In the field of real estate, the idea of predicting the "right price" for a property is growing heavily in interest. Most current algorithms solely use statistical information about given properties as a form of input to predict its right price. However, these algorithms fail to include a notable form of data that often influences the perception of a buyer: visual data of the house. Recently, convolutional neural networks (CNNs) have increased in prominence for their ability to generate strong feature representations out of images and use those representations to accurately map visual inputs to scalar/vectorized outputs. 

Our goal was to create a custom convolutional neural network to accurately predict Toronto housing prices with less than 20% error.
## Features and Overview
* Combined CNN and dense network model
  * Easy to swap CNN model architectures
  * Easy to change dense network size
* Transfer learning using California and Toronto housing datasets
  * [California Dataset](https://github.com/emanhamed/Houses-dataset)
  * Also modified dataset to use latitude/longitude values in place of postal codes
* Custom Toronto Dataset we collected in February 2021 (157 houses)
* Image data augmentation (crop, rotate, mirroring, saturation, brightness)
  * Inputs to Network:
  * 2x2 Mosaic image (bedroom, bathroom, kitchen, frontal view)
  * Price, Number of bedrooms, bathrooms, square feet, and postal code
* Configurable training (hyperparameters, model architecture) with `config.yaml`
## Installation and Quick Setup
To download our code:
```git clone “url”```

Dependent Packages:
`tensorflow 2.3.0, matplotlib, opencv-python, numpy, pandas, keras, sklearn`

To install the dependent packages, run:
```bash
pip install -r requirements.txt
```
## Dataset
### California Dataset 
We initially trained our network on a dataset of California houses, created by [Ahmed and Moustafa](https://github.com/emanhamed/Houses-dataset), consisting of both structured data (statistical property information in tabular form) and unstructured data (images). This dataset contains information for over 500 houses, each with 4 images of a bathroom, bedroom, kitchen, and frontal view. Statistical information for each house includes the number of bedrooms, bathrooms, square footage, postal code, and price.

### Toronto Dataset
We created our own Toronto real estate dataset by compiling the images, prices, number of rooms, surface areas, and postal codes of houses on the Toronto Regional Real Estate Board (TRREB) website. A Python script was used to accurately calculate the area of the house from provided measurements of each room. Like the California dataset, in our Toronto dataset, we had four images for each house for the frontal view, bedroom, bathroom, and kitchen. 
## Advanced Configuration
### Training the model with `config.yaml` and `models/`
Our model’s hyperparameters are stored in a config.yaml file. To start training, modify the `config.yaml` if needed and issue the following command
```bash
python pipeline.py
```
#### Import mode `True` vs `False`
Since data augmentation can take considerable time, we can set the `import_mode` in `config.yaml` to skip augmentation to start training immediately. 

On the first run, set `import_mode: False` in `config.yaml` to perform data augmentation. On future runs, you can set `import_mode: True` to skip data augmentation and use previous augmented data. You can always use `import_mode: False` without issues; it just might be slower.

Note: If you switch/modify the dataset or augmentation multiplier, make sure to use `import_mode: True` for the first run.


#### Adding/choosing different model architectures and other hyperparameters
To change hyperparameters like learning rate, optimizer, etc change the parameters on the corresponding lines in the `config.yaml`

In particular, the CNN model and dense model layers are set by the following lines

```yaml
# Train using RegNet as CNN and a 2 layer dense network (8 units in first layer, 4 units in second layer)
CNN_model: 'RegNet'
dense_model:
  - 8
  - 4
```

The number of dense layers and their size can be changed using `config.yaml`. 

Changing the CNN network is more involved, but still straightforward. If you want to add your `CustomNet`, follow the instructions below. As a basic working example, check out how we defined `LeNet` as a CNN in `models/CNN_models/lenet.py` and then used it in `get_network()` in `models/__init__.py`.
Define a function that returns your custom CNN as a [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) in a new file at `models/CNN_models/CustomNet.py` 
Modify `get_network()` in `models/__init__.py` to call your new function with your custom CNN
Change your `config.yaml` to have `CNN_model: 'CustomNet'
### Initial training on California Dataset
To train on the California dataset, specify `directory: 'raw_dataset'` in the first line of the `config.yaml` file. The California dataset is located in the `raw_dataset` directory.
### Transfer learning on Toronto Dataset
To apply transfer learning on the Toronto Dataset, specify `directory: 'toronto_raw_dataset'` in the first line of the `config.yaml` file. The Toronto dataset is located in the `toronto_raw_dataset` directory. 

Remember to set `import_mode: False` in between switching datasets.
## Results
We achieved a test error of 23% using a Zip Code approach on the California dataset, and a test error of 17% using a Latitude and Longitude approach.

The Zip Code accuracy is nearly 4% better compared to contemporary approaches such as https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/. 


