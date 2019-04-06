import numpy as np
import matplotlib.pyplot as plt
import os
import random
from skimage import data
from skimage import transform
from skimage.color import rgb2gray


## Load all plant images from data directory
def load_data(data_directory):
    ## Read all subdirectories in the data directory
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]

    ## Initialize image and label arrays
    images = []
    labels = []

    ## For every subdirectory in the data directory,
    ## read all jpg images name and append it to the images array,
    ## and append subdirecoty name to the labels array.
    for d in directories:
        plant_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(plant_directory, filename)
                      for filename in os.listdir(plant_directory)
                      if filename.endswith(".jpg")]

        for filename in file_names:
            images.append(data.imread(filename))
            labels.append(int(d))

    return images, labels

## Load the images and labels arrays
images, labels = load_data("data")

## Resizes all images in the array to 64x64 px
images64 = [transform.resize(image, (64, 64)) for image in images]
images64_array = np.array(images64)

## Create validation data
sample_indexes = random.sample(range(len(images64_array)), 3720)
validation_images = [images64_array[i] for i in sample_indexes]
validation_labels = [labels[i] for i in sample_indexes]

## Create tensorflow placeholders
imgs = tf.placeholder(dtype = tf.float32, shape = [None, 64, 64])
lbls = tf.placeholder(dtype = tf.int32, shape = [None])

## Flatten layer
images_flat = tf.contrib.layers.flatten(imgs)

## One fully connected layer
fc = tf.contrib.layers.fully_connected(images_flat, 120, tf.nn.relu)

## Loss
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = lbls, logits = fc))

## Optimizer
op = tf.train.AdamOptmizer(learning_rate=0.001).minimize(loss)

## Calculate accuracy
correct_prediction = tf.argmax(fc, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.set_random_seed(1234)

## Start tensorflow session
session = tf.Session()

session.run(tf.global_variables_initializer())
for i in range(801):
    print('EPOCH', i)
    _, acc_val = sess.run([op, accuracy], feed_dict={x: images64, y: labels})
    if i % 10 == 0:
        print("Loss: ", loss)
    print('DONE WITH EPOCH')

## Create and display prediction graph
fig = plt.figure(figsize=(10, 10))

for i in range(len(validation_labels)):
    truth = validation_labels[i]
    prediction = predicted[i]
    plt.subplots(5, 2, 1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(150, 40, "Truth:		{0}\nPrediction:	{1}".format(truth, prediction), fontsize=12, color=color)
    plt.imshow(validation_labels[i], cmap="gray")

plt.show
