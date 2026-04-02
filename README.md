# MLOps Project: MNIST Handwritten Digit Classifier

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-24.0-blue.svg)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Render-Deployed-purple.svg)](https://render.com)

## Project Description

In this project, I demonstrated an end-to-end Machine Learning Operations (MLOps) project that shows the complete stages of a Machine Learning system. This project executes a **handwritten digit classifier** using a Convolutional NeuraL Network (CNN) trained on the **MNIST dataset**. This includes: 

-  Model training and evaluation (99.2% accuracy)
-  REST API for real-time predictions
-  Retraining capability with new data
-  Interactive web UI with visualizations
-  Docker containerization
-  Load testing with Locust
-  Cloud deployment on Render

##  Key Features

| Feature | Description |
|---------|-------------|
| **Digit Prediction** | Upload images of handwritten digits and get real-time predictions with confidence scores (99%+ accuracy) |
| **Model Retraining** | Upload new digit images to improve the model with your own handwriting samples |
| **Data Visualizations** | Explore dataset insights, class distributions, and model performance metrics |
| **Model Uptime** | Track how long the model has been serving predictions |
| **Docker Support** | Containerized application for consistent deployment across environments |
| **Load Testing** | Performance testing with Locust to measure system capacity |

##  Live Demo

The application is deployed and running on Render:

**Live URL**: [https://mlops-project-4lti.onrender.com](https://mlops-project-4lti.onrender.com)

> **Note**: The free tier may take 30-60 seconds to wake up after inactivity.

##  Setup Instructions

### Prerequisites

- Python 3.11 or higher
- Docker Desktop (for containerized deployment)
- Git

### Local Development

1. **Clone the repository**

```bash
git clone https://github.com/Muen1/mlops_project.git
cd mlops_project
```

2. **Create and activate virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

4. Train the model 

```bash
jupyter notebook notebook/model_training.ipynb
# Run all cells to train and save the model
```

5. Run the flask application

```bash
python app/app.py
```

6. Open your browser

[http://localhost:5000](http://localhost:5000)

## Docker Deployment

### Build and run with docker

```bash
# Build the Docker image
docker build -t mlops_project .

# Run the container
docker run -d -p 5000:5000 --name mlops_app mlops_project

# Check logs
docker logs mlops_app

# Stop the container
docker stop mlops_app
```

### Run with Docker compose

```bash
# Start the application
docker-compose up -d

# Scale to multiple containers
docker-compose up -d --scale app=4

# Stop all containers
docker-compose down
```

## Load Test Results

I used **Locust** to evaluate how the system performs under different concurrent user loads on a single Docker container.

### Test Configuration

-**Tool**: Locust 2.17.0

-**Container**: Single Docker container

-**Test Duration**: 60 seconds per test

-**Target Endpoint**: POST /predict (digit prediction)

The POST /predict endpoint is the most resource-intensive because it:
1. Loads the TensorFlow model (100MB+)
2. Processes the uploaded image
3. Runs neural network inference
4. Returns the JSON prediction

The container had failures because with 5+ concurrent users, it could not process all requests within the timeout window. However, the GET endpoints (homepage, visualizations) show 0% failures under all loads.

1. Open new terminal and show locust

```bash
locust -f locustfile.py --host=http://localhost:5000
```

2. Open browser to; 

```text
http://localhost:8089
```

3. Set users and Spawn rate and click start

### Results Summary

| Concurrent Users | Total Requests | Failures | Failure Rate | Avg Response Time | RPS |
|------------------|----------------|----------|--------------|-------------------|-----|
| 3 users | ~150 | ~13 | **9%** | 55ms | 2.5 |
| 5 users | ~174 | 26 | **15%** | 65ms | 2.9 |
| 10 users | ~2982 | 954 | **32%** | 85ms | 6.1 |

### Analysis

The POST `/predict` endpoint is the most resource-intensive because it:

1. Loads the TensorFlow model (100MB+)
2. Processes the uploaded image
3. Runs neural network inference
4. Returns the JSON prediction

**Key Findings:**

| Users | Failure Rate | Status |
|-------|--------------|--------|
| 2 users | 0% |  Optimal capacity |
| 3 users | 9% |  First signs of overload |
| 5 users | 15% |  Moderate overload |
| 10 users | 32% |  Severe overload |

> **Note:** The GET endpoints (homepage, visualizations) showed **0% failures** under all loads.


### Recommendations

1. **Horizontal Scaling**: Deploy 2-4 containers to handle 10+ users (expected 0% failures)
2. **Add Queue**: Use Redis/RabbitMQ for async processing of predictions
3. **Optimize Model**: Use TensorFlow Lite for faster inference (reduces response time by 40%)
4. **Increase Resources**: Use larger container with more CPU/memory allocation.

## How to test the Application

The repository includes test images and retraining data in the `data/` folder:

1. Start the Application (local or Docker)
```bash
# Local
   python app/app.py
   
   # OR Docker
   docker run -d -p 5000:5000 --name mlops_app mlops_project
```

2. Open your browser and go to
[http://localhost:5000](http://localhost:5000)

3. Test Prediction

- Click "Predict a Digit"
- Click "Browse Files"
- Navigate to data/test/ folder
- Select any image (e.g., digit_5.png)
- Click "Predict"

You should see: Prediction: 5 with 99% confidence

4. Test all digits

```bash
# Test all 10 digits with curl
for i in {0..9}; do
  curl -X POST -F "file=@data/test/digit_${i}.png" http://localhost:5000/predict
  echo ""
done
```
5. Testing retraining

- Go to retraining page: http://localhost:5000/retrain

- Upload retraining data:
    - Click "Browse Files"
    - Navigate to data/retrain/retrain_data.zip
    - Click "Upload and Retrain Model"
- Wait for retraining to complete (1-2 minutes)
- Verify retraining worked:
- Go back to prediction page
- Upload the same test image
- The model should still predict correctly

## Project Structure

```text
mlops_project/
├── app/                          # Flask web application
│   ├── __init__.py
│   ├── app.py                    # Main Flask application
│   └── templates/                # HTML templates
│       ├── index.html            # Homepage with uptime
│       ├── predict.html          # Prediction interface
│       ├── visualizations.html   # Data insights
│       └── retrain.html          # Retraining interface
│
├── src/                          # Core ML modules
│   ├── preprocessing.py          # Image preprocessing
│   ├── model.py                  # Model loading with caching
│   ├── prediction.py             # Prediction logic
│   └── retrain.py                # Retraining functionality
│
├── models/                       # Trained model
│   └── mnist_cnn.h5              # CNN model file (99.2% accuracy)
│
├── notebook/                     # Jupyter notebooks
│   └── model_training.ipynb      # Model training and evaluation
|   └── .ipynb_checkpoints/
|       └── model_training-checkpoint.ipynb
│
├── data/                         # Data files
│   ├── train/                    # Training data documentation
│   ├── test/                     # Test images for predictions
│   ├── retrain/                  # Retraining data (zip files)
│   └── load_test_results/        # Load test HTML reports
│
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose configuration
├── locustfile.py                 # Load testing script
├── requirements.txt              # Python dependencies
└── README.md                     
```


## Video Demo

[https://youtu.be/SiKMsCQ26CQ](https://youtu.be/SiKMsCQ26CQ)


## Note on Free Tier Limitations

The Render free tier has 512MB RAM, which is insufficient for the retraining operation (requires ~1GB).

* Prediction endpoint: Works perfectly on both local and cloud deployments

* Retraining: Works on local Docker (more resources)

* For demonstration purposes: Retraining is shown using the local Docker container in the video demo

This demonstrates both the cloud deployment capability and the retraining feature working as designed.

## Dependencies

```text
tensorflow==2.15.0      # Deep learning framework
numpy==1.24.3           # Numerical computing
matplotlib==3.7.2       # Visualization
scikit-learn==1.3.0     # Metrics and utilities
seaborn==0.12.2         # Statistical visualization
pandas==2.0.3           # Data manipulation
flask==2.3.3            # Web framework
pillow==10.0.1          # Image processing
gunicorn==21.2.0        # Production WSGI server
locust==2.17.0          # Load testing
jupyter==1.0.0          # Notebook environment
```
