from flask import Flask
from flask import jsonify
from flask import request
import io
from PIL import Image
import numpy as np
import base64
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

def get_model():
    global model
    model = keras.models.load_model("pneum.h5")
    print("Model Loaded")


def preprocess_image(img, target_size):
    if img.mode!="RGB":
        img=img.convert("RGB")
    img=img.resize(target_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')
    return x

print("Loading Keras Model")
get_model()

@app.route('/check', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded =  base64.b64decode(encoded)
    img = Image.open(io.BytesIO(decoded))   
    processed_image = preprocess_image(image, target_size=(224,224))
    prediction=model.predict(preprocess_image)


    response={
        'result':{  
            'pneumonia':np.argmax(prediction)
        }
    }

    return jsonify(response)
