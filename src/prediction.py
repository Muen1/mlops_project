"""
Prediction module for making digit predictions on images.
"""

import numpy as np
from .model import load_model
from .preprocessing import preprocess_image

# Cache the model
_model = None

def predict(image_path):
    """
    Predict the digit in an image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (predicted_class, confidence_score)
            - predicted_class: int (0-9)
            - confidence_score: float (0-1)
    """
    global _model
    
    # Load model if not already loaded
    if _model is None:
        _model = load_model()
    
    # Preprocess the image
    img_array = preprocess_image(image_path)
    
    # Make prediction
    predictions = _model.predict(img_array, verbose=0)
    
    # Get the predicted class and confidence
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return predicted_class, confidence

def predict_batch(image_paths):
    """
    Predict digits for multiple images.
    
    Args:
        image_paths (list): List of image file paths
        
    Returns:
        list: List of (predicted_class, confidence) tuples
    """
    global _model
    
    if _model is None:
        _model = load_model()
    
    results = []
    for path in image_paths:
        try:
            pred_class, confidence = predict(path)
            results.append({
                'path': path,
                'prediction': pred_class,
                'confidence': confidence,
                'success': True
            })
        except Exception as e:
            results.append({
                'path': path,
                'error': str(e),
                'success': False
            })
    
    return results

def get_top_k_predictions(image_path, k=3):
    """
    Get top k predictions with their confidence scores.
    
    Args:
        image_path (str): Path to the image file
        k (int): Number of top predictions to return
        
    Returns:
        list: List of (class, confidence) tuples for top k predictions
    """
    global _model
    
    if _model is None:
        _model = load_model()
    
    img_array = preprocess_image(image_path)
    predictions = _model.predict(img_array, verbose=0)[0]
    
    # Get top k indices
    top_k_indices = np.argsort(predictions)[-k:][::-1]
    
    results = []
    for idx in top_k_indices:
        results.append((int(idx), float(predictions[idx])))
    
    return results