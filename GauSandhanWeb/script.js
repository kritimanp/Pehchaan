// This is the public URL of your Hugging Face Space
const API_URL = "https://kritssp-pechaan.hf.space/predict/";

// Get references to all our HTML elements
const predictButton = document.getElementById("predictButton");
const imageInput = document.getElementById("imageInput");
const resultsElement = document.getElementById("results");
const loadingElement = document.getElementById("loading");
const resultsContainer = document.getElementById("resultsContainer");

const webcamButton = document.getElementById("webcamButton");
const captureButton = document.getElementById("captureButton");
const videoFeed = document.getElementById("videoFeed");
const canvas = document.getElementById("canvas");

// --- Webcam Logic ---

let stream = null; // Variable to store the webcam stream

webcamButton.addEventListener("click", async () => {
    if (stream) {
        // If stream exists, stop it
        stream.getTracks().forEach(track => track.stop());
        videoFeed.classList.add("hidden");
        captureButton.classList.add("hidden");
        webcamButton.textContent = "Start Webcam";
        stream = null;
    } else {
        // If no stream, start it
        try {
            // Request webcam access
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { facingMode: "environment" } // Prioritize back camera
            });
            videoFeed.srcObject = stream;
            videoFeed.classList.remove("hidden");
            captureButton.classList.remove("hidden");
            webcamButton.textContent = "Stop Webcam";
        } catch (error) {
            console.error("Error accessing webcam:", error);
            resultsContainer.classList.remove("hidden");
            resultsElement.textContent = "Error: Could not access webcam. Please check permissions.";
        }
    }
});

// --- Capture Photo Logic ---

captureButton.addEventListener("click", () => {
    // Set canvas size to match video feed
    canvas.width = videoFeed.videoWidth;
    canvas.height = videoFeed.videoHeight;
    
    // Draw the current video frame onto the canvas
    const context = canvas.getContext("2d");
    context.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
    
    // Convert the canvas drawing to a file (Blob)
    canvas.toBlob(blob => {
        // Create FormData and add the file
        const formData = new FormData();
        formData.append("file", blob, "webcam-capture.png");
        
        // Call the API
        callAPI(formData);
        
        // Stop the webcam stream
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            videoFeed.classList.add("hidden");
            captureButton.classList.add("hidden");
            webcamButton.textContent = "Start Webcam";
            stream = null;
        }
    }, "image/png"); // Specify PNG format
});


// --- File Upload Logic ---

predictButton.addEventListener("click", () => {
    const file = imageInput.files[0];
    if (!file) {
        resultsContainer.classList.remove("hidden");
        resultsElement.textContent = "Please select an image file first.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    
    // Call the API
    callAPI(formData);
});


// --- Reusable API Call Function ---

function callAPI(formData) {
    // Show loading message and clear old results
    resultsContainer.classList.remove("hidden");
    loadingElement.classList.remove("hidden");
    resultsElement.classList.add("hidden");
    resultsElement.textContent = "";

    // Use 'fetch' to send the image to your API
    fetch(API_URL, {
        method: "POST",
        body: formData,
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
    loadingElement.classList.add("hidden");
    resultsElement.classList.remove("hidden");

    if (data.detections && data.detections.length > 0) {
        // backend is already sorted by confidence
        const best = data.detections[0];
        const confidence = (best.confidence * 100).toFixed(2);
        resultsElement.textContent = `Breed: ${best.breed}\nConfidence: ${confidence}%`;
    } else {
        resultsElement.textContent = "No breeds detected in the image.";
    }
})
    .catch(error => {
        loadingElement.classList.add("hidden");
        resultsElement.classList.remove("hidden");
        resultsElement.textContent = `Error: ${error.message}`;
    });
}