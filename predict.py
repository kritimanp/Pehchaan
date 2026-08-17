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


# --- 2. PREPROCESS_IMAGE FUNCTION (letterboxing + normalization) ---

def preprocess_image(image_bytes, imgsz=640):
    """Takes image bytes, resizes and normalizes it for YOLOv8. Returns the
    tensor plus the letterbox params needed to map boxes back to the
    original image size."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    orig_width, orig_height = img.size

    # YOLOv8 preprocessing (letterboxing)
    scale = min(imgsz / orig_width, imgsz / orig_height)
    new_width, new_height = int(orig_width * scale), int(orig_height * scale)
    img_resized = img.resize((new_width, new_height), Image.BILINEAR)

    # Create a new image with padding
    pad_x = (imgsz - new_width) // 2
    pad_y = (imgsz - new_height) // 2
    new_img = Image.new('RGB', (imgsz, imgsz), (114, 114, 114))
    new_img.paste(img_resized, (pad_x, pad_y))

    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(new_img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch

    # Normalize pixel values from 0-255 to 0.0-1.0
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = img_tensor / 255.0

    letterbox_info = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "orig_width": orig_width,
        "orig_height": orig_height,
    }

    return img_tensor, letterbox_info


def unletterbox_box(x1, y1, x2, y2, letterbox_info, imgsz=640):
    """Maps a box from 640x640 letterboxed-image coordinates back to the
    original uploaded image's pixel coordinates."""
    scale = letterbox_info["scale"]
    pad_x = letterbox_info["pad_x"]
    pad_y = letterbox_info["pad_y"]

    # box coords here are still in the 0-1 normalized YOLO output space
    # scaled to imgsz first, then unpadded and unscaled
    x1 = (x1 * imgsz - pad_x) / scale
    y1 = (y1 * imgsz - pad_y) / scale
    x2 = (x2 * imgsz - pad_x) / scale
    y2 = (y2 * imgsz - pad_y) / scale

    # clip to the original image bounds
    x1 = max(0.0, min(x1, letterbox_info["orig_width"]))
    y1 = max(0.0, min(y1, letterbox_info["orig_height"]))
    x2 = max(0.0, min(x2, letterbox_info["orig_width"]))
    y2 = max(0.0, min(y2, letterbox_info["orig_height"]))

    return x1, y1, x2, y2


# --- 3. PREDICT_BREED FUNCTION (with NMS) ---

def predict_breed(image_bytes, conf_threshold=0.5, iou_threshold=0.45, max_detections=20):
    """Runs the full prediction pipeline: preprocess -> inference -> NMS ->
    coordinate unletterboxing. Returns one clean detection per real object
    instead of every raw grid cell above the confidence threshold."""

    # 1. Preprocess the image
    processed_image, letterbox_info = preprocess_image(image_bytes)

    # 2. Run inference
    yolo_output = infer(processed_image)

    # 3. Get the output tensor and convert to a numpy array
    predictions_tensor = yolo_output['output_0'][0]
    predictions_numpy = predictions_tensor.numpy()

    # Transpose from (4+num_classes, 8400) to (8400, 4+num_classes)
    predictions = predictions_numpy.T

    # Split into box coords and class probabilities
    boxes_xywh = predictions[:, 0:4]
    class_probs = predictions[:, 4:]

    class_ids = np.argmax(class_probs, axis=1)
    confidences = np.max(class_probs, axis=1)

    # Convert [x_center, y_center, w, h] -> [x1, y1, x2, y2]
    x_c, y_c, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = x_c - (w / 2)
    y1 = y_c - (h / 2)
    x2 = x_c + (w / 2)
    y2 = y_c + (h / 2)

    # tf.image.non_max_suppression expects boxes as [y1, x1, y2, x2]
    boxes_for_nms = np.stack([y1, x1, y2, x2], axis=1)

    # 4. Non-max suppression: collapses overlapping boxes on the same
    # object down to a single best detection, instead of returning every
    # raw grid cell above the confidence threshold.
    selected_indices = tf.image.non_max_suppression(
        boxes_for_nms,
        confidences,
        max_output_size=max_detections,
        iou_threshold=iou_threshold,
        score_threshold=conf_threshold,
    ).numpy()

    # 5. Build clean results, mapping boxes back to original image coordinates
    results = []
    for i in selected_indices:
        ox1, oy1, ox2, oy2 = unletterbox_box(x1[i], y1[i], x2[i], y2[i], letterbox_info)

        results.append({
            "breed": CLASS_NAMES[int(class_ids[i])],
            "confidence": float(confidences[i]),
            "bbox": [float(ox1), float(oy1), float(ox2), float(oy2)],
        })

    # Sort by confidence, highest first
    results.sort(key=lambda d: d["confidence"], reverse=True)

    return {"detections": results}