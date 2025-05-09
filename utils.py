from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import cv2
import os
import time
from datetime import datetime

# Setup simple logging
log_file = f"logs/utils_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_message(message):
    """Simple logging function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    with open(log_file, 'a') as f:
        f.write(log_entry + "\n")

# Fungsi untuk memproses gambar dengan beberapa metode berbeda
def process_image(file_path, target_size=(224, 224)):
    """
    Memproses gambar untuk prediksi model dengan target size 224x224
    """
    log_message(f"Processing image: {file_path}")
    start_time = time.time()
    
    # Metode 1: Menggunakan Keras image preprocessing (standar)
    try:
        log_message("Trying Method 1: Keras preprocessing")
        img = image.load_img(file_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0  # Normalisasi
        img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi untuk batch
        log_message(f"Method 1 successful - Shape: {img_array.shape}, Range: {img_array.min():.4f}-{img_array.max():.4f}")
        
        # Save a sample preprocessed image for debugging
        sample_path = f"logs/processed_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        np.save(sample_path, img_array)
        log_message(f"Saved preprocessed sample to {sample_path}")
        
        log_message(f"Image processing completed in {time.time() - start_time:.2f} seconds")
        return img_array
        
    except Exception as e:
        log_message(f"Method 1 failed: {e}")
        
        # Metode 2: Menggunakan OpenCV (alternatif)
        try:
            log_message("Trying Method 2: OpenCV preprocessing")
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"Failed to load image with OpenCV: {file_path}")
                
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV menggunakan BGR, konversi ke RGB
            img_array = img.astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            log_message(f"Method 2 successful - Shape: {img_array.shape}, Range: {img_array.min():.4f}-{img_array.max():.4f}")
            
            log_message(f"Image processing completed in {time.time() - start_time:.2f} seconds")
            return img_array
            
        except Exception as e:
            log_message(f"Method 2 failed: {e}")
            
            # Metode 3: Fallback dengan pilihan berbeda
            try:
                log_message("Trying Method 3: Fallback with different approach")
                # Load dengan PIL directly sebagai fallback
                from PIL import Image
                
                img = Image.open(file_path).resize(target_size)
                img_array = np.array(img).astype(np.float32) / 255.0
                
                # Cek apakah gambar grayscale, jika ya, konversi ke RGB
                if len(img_array.shape) == 2:
                    log_message("Converting grayscale to RGB")
                    img_array = np.stack((img_array,) * 3, axis=-1)
                    
                # Jika gambar memiliki alpha channel, buang channel tersebut
                if img_array.shape[-1] == 4:
                    log_message("Removing alpha channel")
                    img_array = img_array[..., :3]
                
                img_array = np.expand_dims(img_array, axis=0)
                log_message(f"Method 3 successful - Shape: {img_array.shape}, Range: {img_array.min():.4f}-{img_array.max():.4f}")
                
                log_message(f"Image processing completed in {time.time() - start_time:.2f} seconds")
                return img_array
                
            except Exception as e:
                log_message(f"Method 3 failed: {e}")
                raise ValueError(f"All image processing methods failed for {file_path}")

# Fungsi untuk mendiagnosa model
def diagnose_model(model, verbose=True):
    """
    Mendiagnosa model untuk membantu debugging
    """
    try:
        log_message("Diagnosing model...")
        
        # Dapatkan info input model
        input_shape = model.input_shape
        log_message(f"Model expects input shape: {input_shape}")
        
        # Dapatkan info output model
        output_shape = model.output_shape
        log_message(f"Model output shape: {output_shape}")
        
        # Cek layer terakhir untuk memastikan aktivasi yang tepat
        last_layer = model.layers[-1]
        log_message(f"Last layer: {last_layer.name}, type: {type(last_layer).__name__}")
        
        # Jika verbose, cetak semua layer
        if verbose:
            log_message("Model architecture:")
            for i, layer in enumerate(model.layers):
                log_message(f"  Layer {i}: {layer.name} - {type(layer).__name__}")
                
        return {
            "input_shape": input_shape,
            "output_shape": output_shape,
            "last_layer": last_layer.name
        }
        
    except Exception as e:
        log_message(f"Error diagnosing model: {e}")
        return None

# Fungsi untuk verifikasi prediksi dengan manual class mapping
def verify_prediction(prediction):
    """
    Memverifikasi dan memformat hasil prediksi dengan manual class mapping
    """
    try:
        log_message(f"Verifying prediction: {prediction}")
        
        # Manual class mapping
        class_mapping = {0: "NO", 1: "YES"}
        
        # Jika prediction adalah array
        if isinstance(prediction, np.ndarray):
            if prediction.ndim > 1 and prediction.shape[1] > 1:
                # Multi-class prediction
                pred_class = np.argmax(prediction, axis=1)[0]
                confidence = prediction[0][pred_class]
                log_message(f"Predicted class index: {pred_class}, confidence: {confidence:.4f}")
            else:
                # Binary prediction
                pred_value = prediction[0][0]
                pred_class = 1 if pred_value > 0.5 else 0
                confidence = pred_value if pred_class == 1 else 1 - pred_value
                log_message(f"Binary prediction: {pred_value:.4f}, class: {pred_class}, confidence: {confidence:.4f}")
        else:
            pred_class = prediction
            confidence = 1.0
            log_message(f"Single prediction value: {pred_class}")
            
        # Convert to label
        if pred_class in class_mapping:
            label = class_mapping[pred_class]
            log_message(f"Final prediction label: {label}")
        else:
            log_message(f"WARNING: Class index {pred_class} not found in manual mapping")
            label = f"Unknown (Class {pred_class})"
            
        return {
            "class_index": int(pred_class),
            "label": label,
            "confidence": float(confidence) * 100  
        }
        
    except Exception as e:
        log_message(f"Error verifying prediction: {e}")
        return {
            "class_index": -1,
            "label": "Error",
            "confidence": 0.0,
            "error": str(e)
        }