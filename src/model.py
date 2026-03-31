import tensorflow as tf
import os

MODEL_PATH = 'models/mnist_cnn.h5'

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)
