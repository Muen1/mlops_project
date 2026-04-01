# Training Data

The model was trained on the **MNIST Handwritten Digit Dataset**, a classic benchmark dataset for image classification.

##  Dataset Overview

- **Dataset**: MNIST (Modified National Institute of Standards and Technology)
- **Type**: Handwritten digit images
- **Training Samples**: 60,000 images
- **Test Samples**: 10,000 images
- **Image Size**: 28 x 28 pixels (grayscale)
- **Classes**: 10 digits (0 through 9)
- **Class Balance**: ~6,000 images per digit (perfectly balanced)

##  Dataset Source

The dataset is built into TensorFlow/Keras and is automatically downloaded when running the training notebook:

```python
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

```

**Original Source:** Yann LeCun's MNIST Database

## **Training Details**

### **Data Preprocessing**

**Normalization:** Pixel values scaled from [0, 255] to [0, 1]

**Reshaping:** Added channel dimension for CNN compatibility (28, 28, 1)

**Train/Validation Split:** 80% training, 20% validation

### **Model Architecture**

The model uses a Convolutional Neural Network (CNN) with:

3 Convolutional layers with ReLU activation

Batch Normalization for training stability

MaxPooling layers for downsampling

Dropout layers (0.25-0.5) to prevent overfitting

L2 Regularization (0.001) for weight constraints

2 Dense layers for final classification

### **Training Optimization**

Optimizer: Adam (learning rate = 0.001)

Loss Function: Categorical Crossentropy

Early Stopping: Patience of 5 epochs

Learning Rate Reduction: Reduce on plateau (factor = 0.5)

Model Checkpoint: Saves best model based on validation accuracy

Epochs: Up to 50 (early stopping typically stops at ~15-20)

## **Model Performance**
After training, the model achieves:

Test Accuracy: 99.2%

Test Precision: 99.1%

Test Recall: 99.1%

Test F1-Score: 99.1%

## **Data Files**

The actual MNIST dataset is not stored in this folder because:

It's automatically downloaded by TensorFlow when needed

The dataset is cached in TensorFlow's internal directory

It's large (11MB) and unnecessary to store in the repository

The model file (models/mnist_cnn.h5) contains all learned weights and can make predictions without the original training data.

## **Reproducibility**

To reproduce training:

Run notebook/model_training.ipynb

TensorFlow will automatically download MNIST

The notebook contains all preprocessing and training steps

Model is saved to models/mnist_cnn.h5

## **References**

LeCun, Y., Cortes, C., & Burges, C. (1998). The MNIST database of handwritten digits.

TensorFlow Documentation: MNIST Dataset
EOF
