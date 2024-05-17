from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from keras.models import load_model as keras_load_model
from keras.models import Sequential
from typing import List
import numpy as np
from PIL import Image
from io import BytesIO
import argparse
import uvicorn
from time import time
import psutil

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram

app = FastAPI()

# Initialize Prometheus metrics
num_requests = Counter('num_requests', 'Number of requests received', ['method', 'endpoint', 'ip_address'])
processing_time_per_char = Gauge('processing_time_per_char', 'Processing time per character in microseconds', ['method', 'endpoint'])
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
network_io_sent = Counter('network_io_sent_bytes_total', 'Total number of bytes sent via network')
network_io_received = Counter('network_io_received_bytes_total', 'Total number of bytes received via network')

# Setup Prometheus instrumentation for FastAPI
Instrumentator().instrument(app).expose(app)

# Global model variable
model: Sequential = None

def load_model(path: str) -> Sequential:
    """Loads a Keras model from the specified path and returns it."""
    global model
    try:
        model = keras_load_model(path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
    return model

def predict_digit(data_point: List[float]) -> str:
    """Uses the loaded model to predict the digit from the image data."""
    if model is None:
        raise ValueError("Model is not loaded.")
    prediction = model.predict(np.array([data_point]))
    return str(np.argmax(prediction))

@app.post("/predict/")
async def predict_api(request: Request, file: UploadFile = File(...)):
    """Endpoint to receive an image and predict the digit using the model."""
    num_requests.labels(method="POST", endpoint="/predict", ip_address=request.client.host).inc()
    
    start_time = time()
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image = image.convert("L")
        image_array = np.asarray(image)
        flattened_image_array = image_array.flatten() / 255.0
        digit = predict_digit(flattened_image_array.tolist())
        
        end_time = time()
        processing_duration = (end_time - start_time) * 1000  # Convert to milliseconds
        input_length = len(flattened_image_array)
        processing_time_per_char_value = (processing_duration / input_length) * 1000  # Convert to microseconds per character
        processing_time_per_char.labels(method="POST", endpoint="/predict").set(processing_time_per_char_value)
        
        # System Metrics Update
        memory_usage.set(psutil.virtual_memory().used)
        cpu_usage.set(psutil.cpu_percent())
        net_io = psutil.net_io_counters()
        network_io_received.inc(net_io.bytes_recv)
        network_io_sent.inc(net_io.bytes_sent)
        
        return {"digit": digit}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Digit Prediction API")
    parser.add_argument("model_path", type=str, default="/app/mnist_model.h5", help="Path to the saved Keras model")
    args = parser.parse_args()
    load_model(args.model_path)
    uvicorn.run(app, host="0.0.0.0", port=8000)

