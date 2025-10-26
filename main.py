# main.py
from fastapi import FastAPI, File, UploadFile
from predict import predict_breed  # Imports your prediction function
import uvicorn

app = FastAPI(title="Pehchaan Breed Detection API")

@app.get("/")
async def root():
    """A simple root endpoint to check if the API is running."""
    return {"message": "Welcome to the Pehchaan API! Ready to detect breeds."}

@app.post("/predict/")
async def create_prediction(file: UploadFile = File(...)):
    """
    Receives an image file and returns breed detections.
    """
    # 1. Read the image file sent by the user
    image_bytes = await file.read()
    
    # 2. Pass the bytes to your prediction function
    detection_results = predict_breed(image_bytes)
    
    # 3. Return the JSON results
    return detection_results

if __name__ == "__main__":
    # This line allows you to run the app by just running `python main.py`
    uvicorn.run(app, host="127.0.0.1", port=8000)