"""
Model loading and management module.
Handles loading the trained model and providing access to it.
"""

import tensorflow as tf
import os
import sys

# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'mnist_cnn.h5')

# Global variable to cache the model
_model = None

def load_model():
    """
    Load the trained model from disk.
    Uses caching to avoid reloading multiple times.
    
    Returns:
        tf.keras.Model: Loaded TensorFlow model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    global _model
    
    if _model is not None:
        return _model
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    
    try:
        print(f"Loading model from {MODEL_PATH}...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return _model
    
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        dict: Model information including architecture and summary
    """
    model = load_model()
    
    info = {
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'total_params': model.count_params(),
        'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
        'layers': [layer.name for layer in model.layers]
    }
    
    return info

def model_exists():
    """
    Check if the model file exists.
    
    Returns:
        bool: True if model exists, False otherwise
    """
    return os.path.exists(MODEL_PATH)