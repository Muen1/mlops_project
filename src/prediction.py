import numpy as np
from .model import load_model
from .preprocessing import preprocess_image

model = None

def predict(image_path):
    global model
    if model is None:
        model = load_model()
    img = preprocess_image(image_path)
    pred = model.predict(img)
    predicted_class = np.argmax(pred, axis=1)[0]
    return predicted_class, pred[0][predicted_class]
