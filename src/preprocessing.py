"""
Image preprocessing module for MNIST digit classification.
Handles loading, resizing, and normalizing images for model prediction.
"""

import numpy as np
from PIL import Image
import os

def preprocess_image(image_path):
    """
    Loads an image, converts to grayscale, resizes to 28x28, and normalizes.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image array of shape (1, 28, 28, 1)
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale (L mode)
        img = img.convert('L')
        
        # Resize to 28x28 pixels
        img = img.resize((28, 28))
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img) / 255.0
        
        # Add batch and channel dimensions
        img_array = np.expand_dims(img_array, axis=(0, -1))
        
        return img_array
    
    except Exception as e:
        raise Exception(f"Error preprocessing image {image_path}: {str(e)}")

def preprocess_image_batch(image_paths):
    """
    Preprocess multiple images for retraining.
    
    Args:
        image_paths (list): List of image file paths
        
    Returns:
        list: List of preprocessed image arrays
    """
    processed_images = []
    for path in image_paths:
        img_array = preprocess_image(path)
        processed_images.append(img_array[0])  # Remove batch dimension
    return processed_images