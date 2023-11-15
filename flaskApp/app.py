from flask import Flask, render_template, request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)

model = load_model('model.h5')

@app.route('/', methods=['GET'])
def hello_world(): 
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = './images/' + imagefile.filename 
    imagefile.save(image_path)
    
    image = load_img(image_path, target_size=(30, 30,3))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    resize_image = image/255

    pred = np.argmax(model.predict(resize_image), axis=-1)

    return render_template('index.html', prediction= str(pred))


if __name__ == '__main__':
    app.run(port = 3000, debug = True)
