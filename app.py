from flask import Flask, redirect, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
model_mobilenet = load_model('MobileNetv2_leaf.h5')
# model_vgg16 = load_model('Vgg16_leaf.h5')
model_resnet = load_model('Resnet_leaf.h5')
model_inceptionv3 = load_model('Inceptionv3_leaf.h5')

def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    prediction = model.predict(img_array)
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    models = {
        'MobileNetV2': model_mobilenet,
        'VGG16': model_vgg16,
        'ResNet': model_resnet
    }
    class_names = ['Health', 'Bacterial leaf blight', 'Brown spot', 'Leaf smut']
    predictions = {}

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('static', 'images', file.filename)
            file.save(file_path)
            
            for model_name, model in models.items():
                prediction = predict_image(file_path, model)
                predicted_class_index = np.argmax(np.array(prediction))
                predicted_class = class_names[predicted_class_index]
                predictions[model_name]= predicted_class

            return render_template('index.html', predictions=predictions, image_path=file_path)
    return render_template('index.html', predictions=None, image_path=None)

if __name__ == '__main__':
    app.run()
