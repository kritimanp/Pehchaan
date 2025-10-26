# predict.py
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import yaml

# --- 1. SET UP ALL PATHS AND LOAD THE MODEL ---

# GET THE ABSOLUTE PATH TO THIS SCRIPT'S DIRECTORY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# JOIN IT WITH YOUR MODEL FOLDER'S NAME
MODEL_DIR = os.path.join(BASE_DIR, 'best_saved_model') 

# LOAD THE MODEL USING THE NEW, ABSOLUTE PATH
model = tf.saved_model.load(MODEL_DIR) 

# Get the model's signature
infer = model.signatures['serving_default']

# LOAD CLASS NAMES FROM THE METADATA FILE
metadata_path = os.path.join(MODEL_DIR, 'metadata.yaml')
with open(metadata_path, 'r') as f:
    metadata = yaml.safe_load(f)
CLASS_NAMES = metadata['names']


# --- 2. THE CORRECTED PREPROCESS_IMAGE FUNCTION ---

def preprocess_image(image_bytes, imgsz=640):
    """Takes image bytes, resizes and normalizes it for YOLOv8."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # YOLOv8 preprocessing (letterboxing)
    width, height = img.size
    scale = min(imgsz / width, imgsz / height)
    new_width, new_height = int(width * scale), int(height * scale)
    img = img.resize((new_width, new_height), Image.BILINEAR)
    
    # Create a new image with padding
    new_img = Image.new('RGB', (imgsz, imgsz), (114, 114, 114))
    new_img.paste(img, ((imgsz - new_width) // 2, (imgsz - new_height) // 2))
    
    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(new_img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    
    # --- THIS IS THE CRITICAL NORMALIZATION FIX ---
    # 1. Convert to float32
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32) 
    # 2. Normalize pixel values from 0-255 to 0.0-1.0
    img_tensor = img_tensor / 255.0 
    # -----------------------------------------------
    
    return img_tensor # Return the normalized tensor


# --- 3. THE CORRECTED PREDICT_BREED FUNCTION ---

def predict_breed(image_bytes):
    """Runs the full prediction pipeline."""
    
    # --- OUR NEW SANITY CHECK ---
    print("\n\n*** RUNNING V3 (NORMALIZATION FIX) ***\n\n")
    # ----------------------------

    # 1. Preprocess the image (this will now be normalized)
    processed_image = preprocess_image(image_bytes)
    
    # 2. Run inference
    yolo_output = infer(processed_image)
    
    # 3. Get the output tensor and convert to a numpy array
    # 3. Get the output tensor and convert to a numpy array
    predictions_tensor = yolo_output['output_0'][0]
    predictions_numpy = predictions_tensor.numpy()

    # --- THIS IS THE FIX ---
    # Transpose the array from (84, 8400) to (8400, 84)
    # Now we can loop over the 8400 detections
    predictions = predictions_numpy.T
    
    results = []

    # 4. Loop over all detections
    for pred in predictions:
        # Get box coordinates [x_center, y_center, width, height]
        box = pred[0:4]
        
        # Get all class probabilities
        class_probs = pred[4:]
        
        # Find the single class with the highest probability
        confidence = np.max(class_probs)
        class_id = np.argmax(class_probs)
        
        # Filter out weak detections
        if confidence > 0.5: 
            
            # Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]
            x_center, y_center, w, h = box
            x1 = x_center - (w / 2)
            y1 = y_center - (h / 2)
            x2 = x_center + (w / 2)
            y2 = y_center + (h / 2)
            
            results.append({
                "breed": CLASS_NAMES[int(class_id)], 
                "confidence": float(confidence),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
            
    return {"detections": results}