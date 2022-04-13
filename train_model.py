# USAGE
# python train_model.py --conf config/config.json

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from pyimagesearch.nn.gesturenet import GestureNet
from pyimagesearch.utils import Conf
from imutils import paths
import numpy as np
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the input configuration file")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class labels
print("[INFO] loading images...")
imagePaths = list(paths.list_images(conf["dataset_path"]))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the image, convert it to grayscale, and resize it to be a
    # fixed 64x64 pixels, ignoring aspect ratio
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# reshape the data matrix so that it explicitly includes a channel
# dimension
data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))

# one-hot encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

# initialize our gesture recognition CNN and compile it
model = GestureNet.build(64, 64, 1, len(lb.classes_))
opt = Adam(lr=conf["init_lr"],
           decay=conf["init_lr"] / conf["num_epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=conf["bs"]),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // conf["bs"],
    epochs=conf["num_epochs"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=conf["bs"])
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# serialize the model
print("[INFO] saving model...")
model.save(str(conf["model_path"]))

# serialize the label encoder
print("[INFO] serializing label encoder...")
f = open(str(conf["lb_path"]), "wb")
f.write(pickle.dumps(lb))
f.close()
