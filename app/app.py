import os
import sys
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# Add the project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
try:
    from src.prediction import predict
    print(" Prediction module loaded")
except Exception as e:
    print(f" Error loading prediction module: {e}")
    traceback.print_exc()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data/retrain', exist_ok=True)

# Uptime counter
start_time = datetime.now()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    uptime = (datetime.now() - start_time).total_seconds() / 60
    return render_template('index.html', uptime=uptime)

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                pred_class, confidence = predict(filepath)
                os.remove(filepath)
                return jsonify({
                    'prediction': int(pred_class),
                    'confidence': float(confidence)
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return render_template('predict.html')

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/retrain', methods=['GET', 'POST'])
def retrain_page():
    if request.method == 'POST':
        return jsonify({'message': 'Retraining functionality coming soon'})
    return render_template('retrain.html')

if __name__ == '__main__':
    print(" Starting Flask app...")
    print(f" Current directory: {os.getcwd()}")
    print(f" Templates directory: {os.path.exists('app/templates')}")
    app.run(host='0.0.0.0', port=5000, debug=True)
