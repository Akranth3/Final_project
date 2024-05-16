import sys
import io
import PIL
import PIL.Image as Image
import PIL.ImageOps  
sys.path.append("..")

from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
import tensorflow as tf
import numpy as np
import uvicorn
from utils.model_definition import build_densenet_based_model
import time
import prometheus_client
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import StreamingResponse


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
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
    model.load_weights("weights/densenet_weights.h5")
    return model

def preprocess_image(image):
    #perform some preprocessing
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.image.resize(image, [256,256])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def predict_label(image, model):
    # Preprocess the image
    
    # Make the prediction
    prediction = model.predict(tf.reshape(image, (1, 256, 256, 3)) )
    
    return prediction
    
@app.get("/metrics/")
def get_metrics():
    return Response(prometheus_client.generate_latest(), media_type="text/plain")

@app.post("/predict/")
async def predict(request:Request, file: UploadFile = File(...)):
    start_time = time.time()
    image = await file.read()
    
    # Get client IP address
    client_ip = request.client.host
    
    # Increment API usage counter
    api_usage_counter.labels(client_ip).inc()

   
    pil_image = PIL.Image.open(io.BytesIO(image))

    image = preprocess_image(pil_image)

    model = load_model()

    masked_img = predict_label(image, model)
    masked_img = masked_img.reshape(256, 256)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Set API processing time gauges
    api_time_gauge.labels(client_ip).set(total_time)
    api_time_per_char_gauge.labels(client_ip).set(total_time / len(file.filename))
    
    image = Image.fromarray((masked_img * 255).astype(np.uint8))

    # Create a BytesIO object and save the PIL image to it
    img_io = io.BytesIO()
    image.save(img_io, 'PNG', quality=70)

    # Seek to the beginning of the BytesIO object
    img_io.seek(0)

    # Return a StreamingResponse to send the image
    return StreamingResponse(img_io, media_type='image/png')
    # return {"masked_image shape": masked_img.shape, "processing_time": total_time, "client_ip": client_ip, "Saved the image to the location": "../Predictions/"+file.filename+".png"}
    # return FileResponse('processed_image.png', media_type='image/png')

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.0", port=8080)
