import tensorflow as tf
import numpy as np
import os
from .model import MODEL_PATH

def retrain(new_images, new_labels):
    """
    new_images: list of image arrays (already preprocessed to (28,28,1))
    new_labels: list of integer labels (0-9)
    This function loads the existing model and continues training on new data.
    """
    # Load existing model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        raise FileNotFoundError("Base model not found.")

    # Prepare data
    X = np.array(new_images)
    y = tf.keras.utils.to_categorical(new_labels, 10)

    # Retrain (a few epochs, careful not to overfit)
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    # Save updated model
    model.save(MODEL_PATH)
    return True
