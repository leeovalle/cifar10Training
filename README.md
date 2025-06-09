# CIFAR-10 Image Classification Project

This project demonstrates the process of building and training a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset.

## Project Goal

The main objective of this project is to train a CNN model to accurately classify the 10 different classes of images present in the CIFAR-10 dataset.

## Dataset

The project utilizes the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Project Steps

1.  **Data Loading and Preprocessing:**
    *   The CIFAR-10 dataset is loaded using `keras.datasets.cifar10.load_data()`.
    *   The data is split into training, validation, and test sets. Initially, 20% of the training data is used for validation. In the second model iteration, the original test set is split into 5000 for validation and 5000 for testing.
    *   Image pixel values are scaled to the range [0, 1] by dividing by 255.
    *   Class labels are converted to one hot encoded vectors using `to_categorical`.

2.  **Initial Model Development (model\_a):**
    *   A sequential CNN model is defined with convolutional layers, ReLU activation, MaxPooling, Flatten, Dense layers, and a Dropout layer.
    *   The model is compiled with the Adam optimizer and categorical crossentropy loss.
    *   ModelCheckpoint and EarlyStopping callbacks are used for saving the best model and preventing overfitting.
    *   The model is trained on the preprocessed data.

3.  **Initial Model Evaluation:**
    *   The trained model is evaluated on the training, validation, and test sets to report loss and accuracy.
    *   Confusion matrix and recall score are calculated for the validation set.

4.  **Improved Model Development (model\_a\_improved):**
    *   A new sequential CNN model is defined with additional layers (including an extra convolutional block) and Batch Normalization layers.
    *   Data augmentation is applied to the training data using `ImageDataGenerator` with random rotations, shifts, and horizontal flips.
    *   ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau callbacks are used for saving the best model, preventing overfitting, and adjusting the learning rate.
    *   The improved model is trained using the augmented data.

5.  **Improved Model Evaluation:**
    *   The improved model is evaluated on the training, validation, and test sets.
    *   Accuracy, confusion matrix, and recall score are calculated for the test set.

## Model Architecture (Improved Model)

The improved CNN model has the following architecture:

*   **Block 1:**
    *   Conv2D (32 filters, 3x3 kernel)
    *   BatchNormalization
    *   ReLU Activation
    *   Conv2D (32 filters, 3x3 kernel)
    *   BatchNormalization
    *   ReLU Activation
    *   MaxPooling2D (2x2 pool size)
    *   Dropout (0.2)
*   **Block 2:**
    *   Conv2D (64 filters, 3x3 kernel)
    *   BatchNormalization
    *   ReLU Activation
    *   Conv2D (64 filters, 3x3 kernel)
    *   BatchNormalization
    *   ReLU Activation
    *   MaxPooling2D (2x2 pool size)
    *   Dropout (0.3)
*   **Block 3:**
    *   Conv2D (128 filters, 3x3 kernel)
    *   BatchNormalization
    *   ReLU Activation
    *   Conv2D (128 filters, 3x3 kernel)
    *   BatchNormalization
    *   ReLU Activation
    *   MaxPooling2D (2x2 pool size)
    *   Dropout (0.4)
*   **Dense Layers:**
    *   Flatten
    *   Dense (512 units)
    *   BatchNormalization
    *   ReLU Activation
    *   Dropout (0.5)
    *   Dense (10 units, for 10 classes)
    *   Softmax Activation

## Findings

The initial model provides a baseline for performance. The improved model incorporates techniques like Batch Normalization, an additional convolutional block, and data augmentation to enhance performance and potentially mitigate overfitting. The evaluation metrics (accuracy, loss, confusion matrix, recall) on the test set demonstrate the final performance of the improved model.

Here are the final reported metrics for the improved model on the test set:

*   **Total loss on test set:** *0.33965516090393066*
*   **Accuracy of test set:** *0.8830000162124634*
*   **The accuracy using the test set:** *0.883*
*   **The confusion matrix using the test set:**

The confusion matrix using the test set:

| **Predicted →** | **Airplane** | **Automobile** | **Bird** | **Cat** | **Deer** | **Dog** | **Frog** | **Horse** | **Ship** | **Truck** |
|-----------------|--------------|----------------|----------|---------|----------|---------|----------|-----------|----------|-----------|
| **Airplane**    | 471          | 5              | 9        | 4       | 2        | 0       | 0        | 2         | 10       | 9         |
| **Automobile**  | 0            | 470            | 0        | 0       | 0        | 0       | 1        | 0         | 2        | 22        |
| **Bird**        | 19           | 2              | 397      | 11      | 12       | 15      | 24       | 3         | 2        | 3         |
| **Cat**         | 8            | 3              | 13       | 350     | 24       | 46      | 38       | 4         | 4        | 13        |
| **Deer**        | 6            | 0              | 15       | 4       | 431      | 2       | 25       | 10        | 0        | 0         |
| **Dog**         | 3            | 0              | 6        | 46      | 11       | 408     | 17       | 16        | 0        | 5         |
| **Frog**        | 2            | 1              | 1        | 3       | 2        | 0       | 497      | 0         | 2        | 1         |
| **Horse**       | 5            | 0              | 3        | 5       | 18       | 3       | 9        | 459       | 0        | 3         |
| **Ship**        | 12           | 6              | 0        | 2       | 0        | 0       | 1        | 0         | 465      | 10        |
| **Truck**       | 2            | 15             | 0        | 0       | 0        | 1       | 1        | 0         | 1        | 467       |
## How to Run

To run this project, you will need a Python environment with the following libraries installed:

*   tensorflow
*   numpy
*   matplotlib
*   sklearn
*   keras (usually included with tensorflow)

You can run the code in a Jupyter Notebook or Google Colab environment. The code cells can be executed sequentially to perform data loading, preprocessing, model training, and evaluation.