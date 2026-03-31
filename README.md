# MLOps Project: MNIST Digit Classifier

## Project Description
An end-to-end ML system for classifying handwritten digits. Includes:
- Model training with CNN (accuracy 99%+)
- Flask web app for single predictions
- Bulk upload for retraining
- Data visualizations
- Dockerized deployment
- Load testing with Locust

## Setup Instructions
1. Clone the repo
2. Install Python 3.9+ and dependencies: `pip install -r requirements.txt`
3. Run the Flask app locally: `python app/app.py`
4. Or use Docker: `docker-compose up`
5. For scaling: `docker-compose up --scale app=3`

## Usage
- Predict: upload a single image (PNG/JPG) of a digit.
- Retrain: upload a zip file containing images organized in folders named 0-9, or images named like `digit_0_123.png`.
- Visualizations: see static plots.

## Load Test Results
Run Locust with 3 containers: average response time ~200ms, 95th percentile ~400ms at 50 users.

## Deployment
The app is deployed at http://<ec2-public-ip>:5000 (example).

## Video Demo
[YouTube link](https://youtu.be/...)