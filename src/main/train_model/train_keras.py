import cv2
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from skimage import data
from skimage import transform
import tensorflow as tf


## Load all plant images from data directory
def load_data(data_dictionary):
    ## Read all subdirectories in the data directory
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]

    ## Initialize image and label array
    training_data = []

    ## For every subdirectory in the data directory,
    ## read all jpg images name and append it to the images array
    for d in directories:
        plant_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(plant_directory, f)
                     for f in os.listdir(plant_directory)
                     if f.endswith(".jpg")]

        for f in file_names:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            training_data.append([np.array(img), np.array(int(d))])

    return training_data

images_and_labels = load_data("data")

## Split training data into image array and label array
training_images = np.array([i[0] for i in images_and_labels]).reshape(-1, 64, 64, 1)
training_labels = np.array([i[1] for i in images_and_labels])

## Initialize keras sequential model
model = Sequential()

## Define model layers
model.add(InputLayer(input_shape=[64,64,1]))
model.add(Conv2D(filters=40, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Conv2D(filters=80, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Conv2D(filters=164, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=5, padding='same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(31, activation='softmax'))
optimizer = Adam(lr=1e-3)

## Compile the model with the layers defined above
model.compile(optimizer=optimizer, loss='spare_categorical_crossentropy', metrics=['accuracy'])

## Un-comment the next line to save a picture of the model structure
#plot_model(model, show_shapes=True, to_file='model_structure.png')

## Create validation data
sample_indexes = random.sample(range(len(training_images)), 3720)
validation_images = [training_images[i] for i in samples_indexes]
validation_labels = [training_labels[i] for i in sample_indexes]
validation_images_reshaped = np.array([i for i in sample_images]).reshape(-1, 64, 64, 1)

## Train model
history = model.fit(x=training_images, y=training_labels, epochs=1300, 
        batch_size=100, validation_data(validation_images_reshaped, validation_labels))

## Save the model in JSON format
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
     json_file.write(model_json)
## Save the model weights in h5 format
model.save_weights("model/model.h5")
model.summary()

## Un-comment the following lines to save the training and testing graph
#plt.figure(1)
#
#plt.subplot(211)
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('Model Accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train','test'], loc='upper left')
#
#plt.subplot(212)
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model Loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train','test'], loc='upper left')
#plt.subplots_adjust(hspace=0.5)
#plt.savefig('model_graph.png')
