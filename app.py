from flask import Flask, redirect, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model = load_model('Vgg16_leaf.h5')  # Load your trained model

def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=(64, 64))  # Adjust target_size as per your model's input requirements
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image data
    prediction = model.predict(img_array)
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('static', 'images', file.filename)
            file.save(file_path)
            prediction = predict_image(file_path, model)
            return render_template('index.html', prediction=prediction, image_path=file_path)
    return render_template('index.html', prediction=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
