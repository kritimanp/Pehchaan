# Pehchaan - Cow Breed Detection

Pehchaan (पहचान) - AI powered web app to identify Indian cow breeds from an image using YOLOv8.

### Live Demo
**Website:** https://pehchaan-breed-detector.netlify.app/
**API:** https://kritssp-pechaan.hf.space
**Hugging Face Space:** https://huggingface.co/spaces/Kritssp/pechaan

### Features
- Upload image or use webcam (back-camera priority)
- Single best breed prediction with confidence score
- YOLOv8 TensorFlow SavedModel with letterboxing + NMS
- FastAPI backend + Vanilla JS frontend
- Auto-deployed: Frontend on Netlify, Backend on HF Spaces

### Tech Stack
- **Model:** YOLOv8, TensorFlow, Python
- **Backend:** FastAPI, Uvicorn, Pillow, tf.image.non_max_suppression
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** Hugging Face Spaces (Docker), Netlify, GitHub


### How It Works
1. `preprocess_image()` - letterboxes to 640x640
2. `model.signatures['serving_default']` - inference
3. `non_max_suppression()` - best box
4. `unletterbox_box()` - maps back to original image

### Run Locally

**Backend:**
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
