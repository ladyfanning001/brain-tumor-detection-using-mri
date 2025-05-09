from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import os
from werkzeug.utils import secure_filename
import traceback
import time
from datetime import datetime


try:
    from utils import process_image, diagnose_model
    print("Successfully imported utils module")
except Exception as e:
    print(f"Error importing utils module: {e}")
    
    from tensorflow.keras.preprocessing import image 
    def process_image(file_path, target_size=(224, 224)):  
        print(f"Using fallback process_image for {file_path}")
        img = image.load_img(file_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array


if not os.path.exists('logs'):
    os.makedirs('logs')

# Setup logging
log_file = f"logs/app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, 'a') as f:
        f.write(log_entry + "\n")

log_message("Starting brain tumor detection application...")

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Configure upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    log_message(f"Created upload folder: {UPLOAD_FOLDER}")

# Configure allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define class names manually
class_names = {0: "NO", 1: "YES"}  
log_message(f"Using fixed class names: {class_names}")

# Load model
MODEL_PATH = "Lenet_model.h5"  # Path ke model
try:
    log_message(f"Loading model from {MODEL_PATH}...")
    start_time = time.time()
    model = load_model(MODEL_PATH)
    log_message(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
    
    # Print model summary for debug
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    log_message("Model summary:\n" + "\n".join(model_summary))
    
    # Diagnose model to understand its structure
    try:
        diagnose_info = diagnose_model(model)
        log_message(f"Input shape from diagnosis: {diagnose_info['input_shape'] if diagnose_info else 'Unknown'}")
    except Exception as e:
        log_message(f"Model diagnosis failed: {e}")
        
except Exception as e:
    log_message(f"ERROR loading model: {e}")
    log_message(traceback.format_exc())
    model = None

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None
    error = None
    
    if request.method == "POST":
        log_message("POST request received")
        
        if 'file' not in request.files:
            log_message("No file part in request")
            error = "No file selected"
            return render_template("index.html", result=result, image_path=image_path, error=error)
        
        file = request.files["file"]
        
        if file.filename == '':
            log_message("No file selected")
            error = "No file selected"
            return render_template("index.html", result=result, image_path=image_path, error=error)
        
        if not allowed_file(file.filename):
            log_message(f"Invalid file type: {file.filename}")
            error = "Invalid file type. Please upload an image (PNG, JPG, JPEG)"
            return render_template("index.html", result=result, image_path=image_path, error=error)
        
        try:
            # Save the file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"  # Add timestamp to prevent overwriting
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            log_message(f"File saved: {file_path}")
            
            # Process the image - penting: gunakan ukuran 224x224
            log_message("Processing image...")
            img_array = process_image(file_path, target_size=(224, 224))
            log_message(f"Image processed: shape={img_array.shape}, range={img_array.min():.2f}-{img_array.max():.2f}")
            
            if model is None:
                log_message("Model not loaded, cannot predict")
                error = "Model not loaded. Please check logs."
                return render_template("index.html", result=result, image_path=image_path, error=error)
            
            # Make prediction
            log_message("Making prediction...")
            pred = model.predict(img_array)
            log_message(f"Raw prediction: {pred}")
            
            # Handle binary output (1 output node with sigmoid)
            if pred.shape[1] == 1:
                log_message("Handling binary prediction")
                pred_value = float(pred[0][0])
                pred_class = 1 if pred_value > 0.5 else 0
                confidence = pred_value if pred_class == 1 else (1 - pred_value)
                log_message(f"Binary prediction: value={pred_value:.4f}, class={pred_class}, confidence={confidence:.4f}")
            else:
                # Handle categorical output (softmax)
                pred_class = np.argmax(pred, axis=1)[0]
                confidence = np.max(pred)
                log_message(f"Categorical prediction: class={pred_class}, confidence={confidence:.4f}")
            
            # Convert to label using manual mapping
            if pred_class == 0:
                result = "NO"
                log_message("Final prediction: No tumor detected")
            elif pred_class == 1:
                result = "YES"
                log_message("Final prediction: Tumor detected")
            else:
                log_message(f"ERROR: Unexpected class index {pred_class}")
                error = f"Error: Unexpected prediction class {pred_class}"
            
            # Log confidence
            confidence_pct = confidence * 100
            log_message(f"Confidence: {confidence_pct:.2f}%")
            
            image_path = file_path
            
            return render_template("index.html", result=result, image_path=image_path, error=error)
            
        except Exception as e:
            log_message(f"ERROR during processing: {e}")
            log_message(traceback.format_exc())
            error = f"Error processing image: {str(e)}"
    
    return render_template("index.html", result=result, image_path=image_path, error=error)

@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint to check if the application is running correctly"""
    status = {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "class_names": class_names,
        "model_path": MODEL_PATH
    }
    return jsonify(status)

if __name__ == "__main__":
    log_message("Starting Flask development server...")
    app.run(debug=True, host='0.0.0.0', port=5001) 