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
    uptime_rounded = round(uptime, 2)
    return render_template('index.html', uptime=uptime_rounded)

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
        import tempfile
        import shutil
        import zipfile
        from werkzeug.utils import secure_filename
        
        # Check if file was uploaded
        if 'zipfile' not in request.files:
            return jsonify({'error': 'No zip file uploaded'}), 400
        
        zip_file = request.files['zipfile']
        
        if zip_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not zip_file.filename.endswith('.zip'):
            return jsonify({'error': 'File must be a ZIP archive'}), 400
        
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, secure_filename(zip_file.filename))
            zip_file.save(zip_path)
            
            # Extract and process images
            extract_dir = os.path.join(temp_dir, 'extracted')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            # Load images and labels
            from src.retrain import retrain_from_folder
            
            # Find the folder structure
            items = os.listdir(extract_dir)
            
            # Check for retrain_data folder or use root
            if 'retrain_data' in items:
                train_folder = os.path.join(extract_dir, 'retrain_data')
            else:
                train_folder = extract_dir
            
            # Retrain the model
            result = retrain_from_folder(train_folder, epochs=3)
            
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return jsonify({
                'message': ' Model retrained successfully!',
                'epochs': result.get('epochs_completed', 3),
                'final_accuracy': f"{result.get('final_accuracy', 0)*100:.2f}%",
                'samples_processed': result.get('samples_processed', 0)
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Retraining failed: {str(e)}'}), 500
    
    # GET request - show the retrain form
    return render_template('retrain.html')

if __name__ == '__main__':
    print("\n" + "="*50)
    print(" Starting MNIST Digit Classifier Flask App")
    print("="*50)
    print(f" Current directory: {os.getcwd()}")
    print(f" Templates directory: {os.path.exists('app/templates')}")
    print(f" Model exists: {os.path.exists('models/mnist_cnn.h5')}")
    print(f" Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("\n Open your browser and go to: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)