# RealValue

## About

In this project, we created a custom neural network to predict housing prices in Toronto.

## Motivation and Goal

In the field of real estate, the idea of predicting the "right price" for a property is growing heavily in interest. Most current algorithms solely use statistical information about given properties as a form of input to predict its right price. However, these algorithms fail to include a notable form of data that often influences the perception of a buyer: images of the house. Recently, convolutional neural networks (CNNs) have increased in prominence for their ability to generate strong feature representations out of images and use those representations to accurately map to scalar/vectorized outputs.

Our goal was to create a custom neural network to accurately predict Toronto housing prices within 20% accuracy.


## Dataset

We used a dataset for California houses, containing X houses. Each house  has four images: a front view of the house, a bedroom, the kitchen, and a bathroom.

### Cleaning / Feature Engineering


## Network Structure

Our network consists of a two merged networks: a CNN for images, and a dense network for numerical data (e.g. price, number of rooms, square feet).
```put image here```


### Training
Our network was first trained on a dataset for California houses.

### Transfer Learning
After training our network on the California dataset, we applied transfer learning to a dataset of Toronto houses which we collected. This dataset contains X houses.

## Problems We Encountered
- Lack of data -> sol : resampling/augmentation
- Overfitting, high mean value percentage -> sol: model/network selection, feature enginenering on statistical dataset
-

## Results

- TBD

## Progress

Project in progress. Check back later for further updates on progress!

Current Best: 38.3%

Arsh: 36.0% \
Matthew: \
Sean: \
Alex: \
Charles: \
Sepehr:
