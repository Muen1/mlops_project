from locust import HttpUser, task, between
import io
import random
from PIL import Image
import numpy as np

class MNISTUser(HttpUser):
    """Simulates a user testing the MNIST digit classifier"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Create a test image when user starts"""
        # Create a simple 28x28 test image (simulating a digit)
        img = Image.new('L', (28, 28), color=128)
        pixels = np.array(img)
        
        # Create a simple pattern (circle) to simulate a digit
        for i in range(28):
            for j in range(28):
                if (i - 14)**2 + (j - 14)**2 < 100:
                    pixels[i, j] = 255
        
        img = Image.fromarray(pixels)
        self.test_image = io.BytesIO()
        img.save(self.test_image, format='PNG')
        self.test_image.seek(0)
    
    @task(3)
    def predict_digit(self):
        """Test the prediction endpoint - most common task"""
        self.test_image.seek(0)
        files = {
            'file': ('digit.png', self.test_image, 'image/png')
        }
        
        with self.client.post('/predict', files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'prediction' in data and 'confidence' in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def view_homepage(self):
        """Test the homepage"""
        with self.client.get('/', catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def view_predict_page(self):
        """Test the predict page"""
        with self.client.get('/predict', catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def view_visualizations(self):
        """Test the visualizations page"""
        with self.client.get('/visualizations', catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

# Alternative: Simple test if you want to test only prediction
class SimplePredictionUser(HttpUser):
    """Simple user that only tests prediction"""
    
    wait_time = between(1, 2)
    
    def on_start(self):
        img = Image.new('L', (28, 28), color=128)
        pixels = np.array(img)
        for i in range(28):
            for j in range(28):
                if (i - 14)**2 + (j - 14)**2 < 100:
                    pixels[i, j] = 255
        img = Image.fromarray(pixels)
        self.test_image = io.BytesIO()
        img.save(self.test_image, format='PNG')
        self.test_image.seek(0)
    
    @task
    def predict(self):
        self.test_image.seek(0)
        files = {'file': ('digit.png', self.test_image, 'image/png')}
        self.client.post('/predict', files=files)
