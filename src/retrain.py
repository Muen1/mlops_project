"""
Retraining module for updating the model with new data.
"""

import tensorflow as tf
import numpy as np
import os
import sys
import zipfile
import tempfile
import shutil
from datetime import datetime
from PIL import Image

# Get project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'mnist_cnn.h5')

def preprocess_single_image(image_path):
    """
    Preprocess a single image for retraining.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image array of shape (28, 28, 1)
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add channel dimension
        img_array = np.expand_dims(img_array, -1)
        
        return img_array
    
    except Exception as e:
        raise Exception(f"Error preprocessing image {image_path}: {str(e)}")

def load_images_from_folder(folder_path):
    """
    Load and preprocess images from a folder for retraining.
    Expects folder structure: digit_class/image_files
    e.g., data/retrain/0/image1.png, data/retrain/1/image2.png
    
    Args:
        folder_path (str): Path to folder containing class subfolders
        
    Returns:
        tuple: (images, labels)
            - images: list of preprocessed image arrays
            - labels: list of integer labels
    """
    images = []
    labels = []
    
    print(f" Scanning folder: {folder_path}")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(folder_path):
        # Check if current folder name is a digit
        folder_name = os.path.basename(root)
        
        # Also check if there's a retrain_data folder
        if folder_name == 'retrain_data':
            continue
            
        if folder_name.isdigit():
            label = int(folder_name)
            print(f"  Found folder '{folder_name}' with {len(files)} files")
            
            # Process all images in this folder
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    file_path = os.path.join(root, file)
                    try:
                        # Preprocess image
                        img_array = preprocess_single_image(file_path)
                        images.append(img_array)
                        labels.append(label)
                    except Exception as e:
                        print(f"     Error processing {file}: {e}")
    
    # If no images found in digit-named folders, try to extract from filenames
    if len(images) == 0:
        print("  No digit-named folders found, trying to extract from filenames...")
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    # Try to extract digit from filename like "digit_5_001.png"
                    import re
                    match = re.search(r'digit[_\-](\d)', file)
                    if match:
                        label = int(match.group(1))
                        file_path = os.path.join(root, file)
                        try:
                            img_array = preprocess_single_image(file_path)
                            images.append(img_array)
                            labels.append(label)
                            print(f"  Found digit {label} in filename: {file}")
                        except Exception as e:
                            print(f"     Error processing {file}: {e}")
    
    return images, labels

def load_images_from_zip(zip_path):
    """
    Load images from a zip file for retraining.
    
    Args:
        zip_path (str): Path to zip file containing images
        
    Returns:
        tuple: (images, labels)
    """
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Load images from extracted folder
        images, labels = load_images_from_folder(temp_dir)
        
        return images, labels
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def retrain(new_images, new_labels, epochs=5, batch_size=32):
    """
    Retrain the existing model with new data.
    
    Args:
        new_images (list): List of preprocessed image arrays
        new_labels (list): List of integer labels (0-9)
        epochs (int): Number of epochs to train
        batch_size (int): Batch size for training
        
    Returns:
        dict: Training history and status
    """
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Base model not found at {MODEL_PATH}")
    
    # Load existing model
    print(f" Loading existing model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Convert to numpy arrays
    X = np.array(new_images)
    y = np.array(new_labels)
    
    # One-hot encode labels
    y_cat = tf.keras.utils.to_categorical(y, 10)
    
    print(f"\n Retraining with {len(X)} new samples...")
    print(f" Class distribution:")
    for i in range(10):
        count = np.sum(y == i)
        if count > 0:
            print(f"   Digit {i}: {count} images")
    
    # Create backup of original model
    backup_path = MODEL_PATH.replace('.h5', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
    tf.keras.models.save_model(model, backup_path)
    print(f" Backup saved to {backup_path}")
    
    # Retrain the model (only a few epochs to avoid overfitting)
    print(f"\n Starting retraining for {epochs} epochs...")
    history = model.fit(
        X, y_cat,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.2
    )
    
    # Save the updated model
    model.save(MODEL_PATH)
    print(f" Updated model saved to {MODEL_PATH}")
    
    return {
        'status': 'success',
        'epochs_completed': len(history.history['loss']),
        'final_loss': history.history['loss'][-1],
        'final_accuracy': history.history['accuracy'][-1],
        'backup_path': backup_path,
        'samples_processed': len(X),
        'class_distribution': np.bincount(y).tolist()
    }

def retrain_from_folder(folder_path, epochs=5):
    """
    Convenience function to retrain from a folder of images.
    
    Args:
        folder_path (str): Path to folder containing class subfolders
        epochs (int): Number of epochs to train
        
    Returns:
        dict: Training results
    """
    print(f"\n Loading images from {folder_path}...")
    images, labels = load_images_from_folder(folder_path)
    
    if len(images) == 0:
        raise ValueError("No valid images found in the folder")
    
    print(f"\n Found {len(images)} images across {len(np.unique(labels))} classes")
    
    # Call retrain and get history
    result = retrain(images, labels, epochs=epochs)
    
    return result

def retrain_from_zip(zip_path, epochs=5):
    """
    Convenience function to retrain from a zip file.
    
    Args:
        zip_path (str): Path to zip file containing images
        epochs (int): Number of epochs to train
        
    Returns:
        dict: Training results
    """
    print(f"\n Loading images from zip file {zip_path}...")
    images, labels = load_images_from_zip(zip_path)
    
    if len(images) == 0:
        raise ValueError("No valid images found in the zip file")
    
    print(f"\n Found {len(images)} images across {len(np.unique(labels))} classes")
    
    return retrain(images, labels, epochs=epochs)

def get_retraining_status():
    """
    Check if retraining is possible and get model information.
    
    Returns:
        dict: Status information
    """
    model_exists = os.path.exists(MODEL_PATH)
    
    return {
        'model_exists': model_exists,
        'model_path': MODEL_PATH if model_exists else None,
        'can_retrain': model_exists,
        'message': "Model ready for retraining" if model_exists else "No base model found. Please train first."
    }