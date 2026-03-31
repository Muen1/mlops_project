import numpy as np
from PIL import Image

def preprocess_image(image_path):
    """Loads an image, converts to grayscale, resizes to 28x28, normalizes."""
    img = Image.open(image_path).convert('L')  # grayscale
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1,28,28,1)
    return img_array
