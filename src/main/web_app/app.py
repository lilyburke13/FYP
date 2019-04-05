import cv2
from flask import Flask, flash, render_template, request
import keras
from keras.models import *
from keras.optimizers import *
import numpy as np
import os
from werkzeug import secure_filename


## Initiate flask application
app = Flask(__name__)

## Define folder to save images uploaded
UPLOAD_DIR = 'static/uploads'
## Define what type of images can be uploaded
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
## Define the flower species array
## This is used to retrieve the name of the species predicted
FLOWER_SPECIES = ["Aponogeton Distachyos","Impatiens Glandulifera","Myosotis Scorpioides","Achillea Millefolium","Alisma Lanceolatum",
                  "Allium Triquetrum","Angelica Sylvestris","Anthriscus Sylvestris","Apium Nodiflorum","Aponogeton Distachyos",
                  "Bellis Perennis","Butomus Umbellatus","Caltha Palustris","Comfrey","Crocosmia",
                  "Dactylorhiza Fuchsii","Digitalis Purpurea","Epilobium Hirsutum","Solanum Dulcamara","Senecio Aquaticus",
                  "Ranunculus Peltatus Penicillatus","Ranunculus Lingua","Senecio Aquaticus","Parnassia Palustris","Nymphaea Alba",
                  "Nasturtium Officinale","Mimulus Moschatus","Mimulus Guttatus","Ficaria Verna","Iris Pseudachorus","Lythrum Salicaria"]

## Set the upload folder path to the app configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

## Class to communicate with the Keras model
class NeuralNetwork(object):

    ## Load the keras model into the application and be ready for use.
    def __init__(self):
        json_file = open('model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("model/model.h5")
        optimizer = Adam(lr=1e-3)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    ## This is the function used when requesting a prediction from the model.
    def predict(self, image):
        prediction = self.model.predict(image)
        return prediction

    ## This function clears the model session, to be ready for next prediction.
    def clear_model(self):
        keras.backend.clear_session()

## This function translate the predicted class to a plant species name.
def get_flower_name(index):
    flower_name = FLOWER_SPECIES[index]
    return flower_name

## This function is used to check if the user's image upload has an allowed extension.
def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

## Initiates the webpage on index.
@app.route('/')
def main():
   return render_template('index.html')

## Display's the tests webpage
@app.route('/tests')
def load_tests():
    return render_template('tests.html')

## This function deals with the upload of user images
## When user submits an image, it saves to the upload folder if extension is allowed
## It calls the model class, process the image in other to match the requirements of the model
## Requests a prediction on the user's image, calculates the probability
## Finally, displays the results on the user's browser at the index page
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')

        if file and allowed_files(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    neural_network = NeuralNetwork()
    img = cv2.imread(UPLOAD_DIR+"/"+filename, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(img, (64, 64))
    prediction_image = resized_image.reshape([-1, 64, 64, 1])
    model_out = neural_network.predict(prediction_image)
    prediction_index = np.argmax(model_out)
    probability = int(model_out[:,prediction_index] * 100)
    prediction = get_flower_name(prediction_index)
    neural_network.clear_model()
    return render_template('index.html', prediction=prediction, filename=filename, probability=probability)

## Python app initiator
if __name__ == '__main__':
   app.run(debug = True)
