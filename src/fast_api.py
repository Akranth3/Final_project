from fastapi import FastAPI, File, UploadFile
from fastapi.responses import CORSResponse
import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.models import Model
import uvicorn
from utils.model_definition import build_densenet_based_model
import sys
import time
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

sys.path.append("../")

app = FastAPI()
app.add_middleware(
    CORSResponse,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define counters for API usage
api_usage_counter = Counter("api_usage", "API usage counter", ["client_ip"])

Instrumentator().instrument(app).expose(app)

# Define gauges for API processing time
api_time_gauge = Gauge("api_time", "API processing time", ["client_ip"])
api_time_per_char_gauge = Gauge("api_time_per_char", "API processing time per character", ["client_ip"])

def load_model():
    model = build_densenet_based_model()
    model.load_weights("/../weights/densenet_weights.h5")
    return model

def preprocess_image(image):
    #perform some preprocessing
    return image

def predict_label(image, model):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make the prediction
    prediction = model.predict(preprocessed_image)
    
    return prediction
    

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Get client IP address
    client_ip = request.client.host
    
    # Increment API usage counter
    api_usage_counter.labels(client_ip).inc()
    
    model = load_model()
    image = preprocess_image(file)
    label = predict_label(image, model)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Set API processing time gauges
    api_time_gauge.labels(client_ip).set(total_time)
    api_time_per_char_gauge.labels(client_ip).set(total_time / len(file.filename))
    
    return {"label": label}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
