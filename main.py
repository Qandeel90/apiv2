from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from keras.models import load_model
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

MODEL = load_model("wheatleaf3.h5")

CLASS_NAMES = ["Healthy", "septoria", "stripe_rust"]

@app.get("/")
async def home():
    return "Hello, I am Home"

@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    img = np.array(Image.open(BytesIO(data)))
    image = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
  
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {"class": predicted_class, "confidence": float(confidence)}


if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)
