 Plant Species’ detection systemin Lough Corrib using Image Processing and Deep Learning
======================================================================================================

Getting Started
---------------
These instructions will make for the successful deployment of the system implementation on a Ubuntu 16.04 OS.

__Software Prerequisites__

The following softwares are required in order to deploy the system:

- OpenCV3
- TensorFlow
- Keras
- Flask

To install the above software follow the respective installation instructions detailed below.

_OpenCV_

OpenCV is an open source library aimed at real-time computer vision. OpenCV3 can be installed on a Linux OS by following the steps in the tutorial link:

https://www.learnopencv.com/install-opencv3-on-ubuntu/

_TensorFlow_

TensorFlow is an open source library used for machine learning applications. TensorFlow can be installed on a Linux OS using the following commands:

First, try installing TensorFlow directly using pip
```
    $ sudo pip install tensorflow==1.12.0
```
If that installation is unsuccessful, try upgrading python versions by following this tutorial link:

https://www.tensorflow.org/install/pip

_Keras_

Keras is an open-source neural-network library written in Python. It is capable of running on top of TensorFlow. Keras can be installed on a Linux OS using the following command:

```
    $ sudo pip install keras==2.1.4
```
_Flask_

Flask is a micro web framework written in python. Flask can be installed on a Linux OS using the following command:

```
    $ sudo pip install flask
```
System Implentation
------------
The following descibes the different software development components all found in the /FYP/src/main directory.

__image_processing__

This directory contains all the image processing scripts used to prepare the raw dataset for training.

- remove_background.py
- resize_images.py

Within the /tools directory, there is also: 
- balance_data.sh
- rename_files.sh
  
Within the /samples directory, there is also:
- create_samples.sh

__train_model__

This directory contains the script for training the Keras CNN model.

- train_keras.py

__tests__

This directory contains the test_images, test_models and test_training sub-directories. In the /test_model directory, the CNN training models for various numbers of classes can be found. 

The most relevant model is our most accurate, and can be found in /trained_model_31_classes:

- model.h5

__web_app__

This directory contains all the code to run our web app online plant classification system.

- app.py

Deployment
----------
In order to deploy the online plant classifier system follow the steps below:

Firstly navigate into the /web_app directory:
```
    $ cd /FYP/src/main/web_app
```
Next run the app.py script
```
    $ python app.py
```
This will launch the app, and display a link to an ip address. Click to open the link, and it will open the browser and display the app web page.

To use the Plant Classification app:
1) Select the 'Browse...' button in the centre of the interface to open your file system.
2) Choose a plant image to classify, before selecting it for upload.
3) Once it has been uploaded, click the 'Submit' button
4) An instant classification prediction for your image will be displayed on the screen.
5) Repeat the steps above for each new image you wish to classify.

Demo Test Case
----------
To test the classifier app, a folder of unseen flower images has been included.

These images can be found in the directory /tests/test_images

To test the app, follow the steps outlined above, and simply choose one of the test images from this folder to be uploaded. Then simply press Submit on the app screen and a prediction result will be returned.