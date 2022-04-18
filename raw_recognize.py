# USAGE
# python raw_recognize.py --conf config/config.json

# import the necessary packages
from Python_Serial_to_Arduino import arduino_serial_write
from pyimagesearch.utils import Conf
# from pyimagesearch.notifications import TwilioNotifier
from imutils.video import VideoStream
from imutils import paths
from threading import Thread
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from datetime import datetime
from datetime import date
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# construct the argument parser and parse the arguments
from text_to_speech import text_to_speech


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the input configuration file")
args = vars(ap.parse_args())

# load the configuration file and initialize the Twilio notifier
conf = Conf(args["conf"])
# tn = TwilioNotifier(conf)

# grab the paths to gesture icon images and then initialize the icons
# dictionary where the key is the gesture name (derived from the image
# path) and the key is the actual icon image
print("[INFO] loading icons...")
imagePaths = paths.list_images(conf["assets_path"])
icons = {}

# loop over the image paths
for imagePath in imagePaths:
    # extract the gesture name (label) the icon represents from the
    # filename, load the icon, and then update the icons dictionary
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    icon = cv2.imread(imagePath)
    icons[label] = icon

# grab the top-left and and bottom-right (x, y)-coordinates for the
# gesture capture area
speech_to_text_enable = str(conf["speechToText"])
arduino_port = str(conf["arduino_port"])

TOP_LEFT = tuple(conf["top_left"])
BOT_RIGHT = tuple(conf["bot_right"])

# load the trained gesture recognizer model and the label binarizer
print("[INFO] loading model...")
print(conf["model_path"])
# model = load_model(str(conf["model_path"]))
model = load_model(str(conf["model_path"]))
lb = pickle.loads(open(str(conf["lb_path"]), "rb").read())

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# initialize the current gesture, a bookkeeping variable used to keep
# track of the number of consecutive frames a given gesture has been
# classified as
currentGesture = [None, 0]

# initialize the list of input gestures recognized from the user
# along with the timestamp of when all four gestures were entered
gestures = []
enteredTS = None

# initialize two booleans used to indicate (1) whether or not the
# alarm has been raised and (2) if the correct pass code was entered
alarm = False
correct = False
flag = 0
on_flag = 0
while_flag = 0
label_count = 0
previous_value = 0

# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream and grab the
    # current timestamp
    frame = vs.read()
    timestamp = datetime.now()

    # resize the frame and then flip it horizontally
    frame = imutils.resize(frame, width=300)
    frame = cv2.flip(frame, 1)

    # clone the original frame and then draw the gesture capture area
    clone = frame.copy()
    cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)

    # only perform hand gesture classification if the current gestures
    # list is not already full
    #    if len(gestures) < 4:
    # extract the hand gesture capture ROI from the frame, convert
    # the ROI to grayscale, and then threshold it to reveal a
    # binary mask of the hand
    roi = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.threshold(roi, 75, 255, cv2.THRESH_BINARY)[1]
    visROI = roi.copy()

    # now that we have the hand region we need to resize it to be
    # the same dimensions as what our model was trained on, scale
    # it to the range [0, 1], and prepare it for classification
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    # classify the input image
    proba = model.predict(roi)[0]
    label = lb.classes_[proba.argmax()]
    print("Current Recognition", label)

    if previous_value == label:
        label_count += 1
        print(label_count)
        if 1 < label_count < 5:
            print("Speech Text", str(label))
            text_to_speech(str(label), speech_to_text_enable)

    else:
        label_count = 0

    previous_value = label
    if label == "one" and flag == 0:

        flag = 1
        print("first room is selected")
        time.sleep(5)
    elif label == "two" and flag == 0:
        flag = 2
        print("second room is selected")
        time.sleep(5)
    if flag == 1:
        while True:
            print("Inside first room")
            frame = vs.read()
            timestamp = datetime.now()

            # resize the frame and then flip it horizontally
            frame = imutils.resize(frame, width=300)
            frame = cv2.flip(frame, 1)

            # clone the original frame and then draw the gesture capture area
            clone = frame.copy()
            cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)

            roi = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.threshold(roi, 75, 255, cv2.THRESH_BINARY)[1]
            visROI = roi.copy()

            # now that we have the hand region we need to resize it to be
            # the same dimensions as what our model was trained on, scale
            # it to the range [0, 1], and prepare it for classification
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # classify the input image
            proba = model.predict(roi)[0]
            label = lb.classes_[proba.argmax()]
            print("In 1st Room, Please Select a Device \n 1. ONE \n 2.TWO \n", label)

            if label == "one" and flag == 1:
                print("first room 1st device is selected")
                on_flag = 1

                while True:

                    frame = vs.read()
                    timestamp = datetime.now()

                    # resize the frame and then flip it horizontally
                    frame = imutils.resize(frame, width=300)
                    frame = cv2.flip(frame, 1)

                    # clone the original frame and then draw the gesture capture area
                    clone = frame.copy()
                    cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)

                    roi = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi = cv2.threshold(roi, 75, 255, cv2.THRESH_BINARY)[1]
                    visROI = roi.copy()

                    # now that we have the hand region we need to resize it to be
                    # the same dimensions as what our model was trained on, scale
                    # it to the range [0, 1], and prepare it for classification
                    roi = cv2.resize(roi, (64, 64))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # classify the input image
                    proba = model.predict(roi)[0]
                    label = lb.classes_[proba.argmax()]
                    print("1st Room - 1st Device \n 1. ON -> Five \n 2. OFF -> Four \n", label)

                    if label == "five" and on_flag == 1:
                        on_flag = 0
                        flag = 0
                        print("first room 1st device on")
                        arduino_serial_write(x=111, port=arduino_port)
                        time.sleep(2)
                        break
                    elif label == "four" and on_flag == 1:
                        on_flag = 0
                        flag = 0
                        print("first room 1st device off")
                        arduino_serial_write(x=110, port=arduino_port)
                        time.sleep(2)
                        break

                    if (flag == 0):
                        break

                    cv2.imshow("ROI", visROI)

                    key = cv2.waitKey(1) & 0xFF

                    time.sleep(1)

            if label == "two" and flag == 1:
                print("first room 2nd device is selected")
                on_flag = 1

                while True:

                    frame = vs.read()
                    timestamp = datetime.now()

                    # resize the frame and then flip it horizontally
                    frame = imutils.resize(frame, width=300)
                    frame = cv2.flip(frame, 1)

                    # clone the original frame and then draw the gesture capture area
                    clone = frame.copy()
                    cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)

                    roi = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi = cv2.threshold(roi, 75, 255, cv2.THRESH_BINARY)[1]
                    visROI = roi.copy()

                    # now that we have the hand region we need to resize it to be
                    # the same dimensions as what our model was trained on, scale
                    # it to the range [0, 1], and prepare it for classification
                    roi = cv2.resize(roi, (64, 64))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # classify the input image
                    proba = model.predict(roi)[0]
                    label = lb.classes_[proba.argmax()]
                    print("1st Room - 2nd Device \n 1. ON -> Five \n 2. OFF -> Four \n", label)

                    if label == "five" and on_flag == 1:
                        on_flag = 0
                        flag = 0
                        print("first room 2nd device on")
                        arduino_serial_write(x=121, port=arduino_port)
                        time.sleep(2)
                        break
                    elif label == "four" and on_flag == 1:
                        on_flag = 0
                        flag = 0
                        print("first room 2nd device off")
                        arduino_serial_write(x=120, port=arduino_port)
                        time.sleep(2)
                        break

                    if (flag == 0):
                        break

                    cv2.imshow("ROI", visROI)

                    key = cv2.waitKey(1) & 0xFF

                    time.sleep(1)

            time.sleep(1)
            if (flag == 0):
                break

    if flag == 2:
        while True:
            print("Inside second room")
            frame = vs.read()
            timestamp = datetime.now()

            # resize the frame and then flip it horizontally
            frame = imutils.resize(frame, width=300)
            frame = cv2.flip(frame, 1)

            # clone the original frame and then draw the gesture capture area
            clone = frame.copy()
            cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)

            roi = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.threshold(roi, 75, 255, cv2.THRESH_BINARY)[1]
            visROI = roi.copy()

            # now that we have the hand region we need to resize it to be
            # the same dimensions as what our model was trained on, scale
            # it to the range [0, 1], and prepare it for classification
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # classify the input image
            proba = model.predict(roi)[0]
            label = lb.classes_[proba.argmax()]
            print("In 2nd Room , Please Select a Device \n 1. ONE \n 2.TWO \n", label)

            if label == "one" and flag == 2:
                print("Second room 1st device is selected")
                on_flag = 1

                while True:

                    frame = vs.read()
                    timestamp = datetime.now()

                    # resize the frame and then flip it horizontally
                    frame = imutils.resize(frame, width=300)
                    frame = cv2.flip(frame, 1)

                    # clone the original frame and then draw the gesture capture area
                    clone = frame.copy()
                    cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)

                    roi = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi = cv2.threshold(roi, 75, 255, cv2.THRESH_BINARY)[1]
                    visROI = roi.copy()

                    # now that we have the hand region we need to resize it to be
                    # the same dimensions as what our model was trained on, scale
                    # it to the range [0, 1], and prepare it for classification
                    roi = cv2.resize(roi, (64, 64))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # classify the input image
                    proba = model.predict(roi)[0]
                    label = lb.classes_[proba.argmax()]
                    print("2nd Room - 1st Device \n 1. ON -> Five \n 2. OFF -> Four", label)

                    if label == "five" and on_flag == 1:
                        on_flag = 0
                        flag = 0
                        print("Second room 1st device on")
                        arduino_serial_write(x=211, port=arduino_port)
                        time.sleep(1)
                        break
                    elif label == "four" and on_flag == 1:
                        on_flag = 0
                        flag = 0
                        print("Second room 1st device off")
                        arduino_serial_write(x=210, port=arduino_port)
                        time.sleep(1)
                        break

                    if (flag == 0):
                        break

                    cv2.imshow("ROI", visROI)

                    key = cv2.waitKey(1) & 0xFF

                    time.sleep(2)

            time.sleep(1)
            if (flag == 0):
                break

            if label == "two" and flag == 2:
                print("Second room 2nd device is selected")
                on_flag = 1

                while True:

                    frame = vs.read()
                    timestamp = datetime.now()

                    # resize the frame and then flip it horizontally
                    frame = imutils.resize(frame, width=300)
                    frame = cv2.flip(frame, 1)

                    # clone the original frame and then draw the gesture capture area
                    clone = frame.copy()
                    cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)

                    roi = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi = cv2.threshold(roi, 75, 255, cv2.THRESH_BINARY)[1]
                    visROI = roi.copy()

                    # now that we have the hand region we need to resize it to be
                    # the same dimensions as what our model was trained on, scale
                    # it to the range [0, 1], and prepare it for classification
                    roi = cv2.resize(roi, (64, 64))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    # classify the input image
                    proba = model.predict(roi)[0]
                    label = lb.classes_[proba.argmax()]
                    print("2nd Room - 2nd Device \n 1. ON -> Five \n 2. OFF -> Four", label)

                    if label == "five" and on_flag == 1:
                        on_flag = 0
                        flag = 0
                        print("Second room 2nd device on")
                        arduino_serial_write(x=211, port=arduino_port)
                        time.sleep(1)
                        break
                    elif label == "four" and on_flag == 1:
                        on_flag = 0
                        flag = 0
                        print("Second room 2nd device off")
                        arduino_serial_write(x=210, port=arduino_port)
                        time.sleep(1)
                        break

                    if (flag == 0):
                        break

                    cv2.imshow("ROI", visROI)

                    key = cv2.waitKey(1) & 0xFF

                    time.sleep(2)

            time.sleep(1)
            if (flag == 0):
                break

    #
    # #    # show ROI we're monitoring, the output frame, and passcode info
    cv2.imshow("ROI", visROI)
    #    cv2.imshow("Security Feed", clone)
    #    cv2.imshow("Passcode", canvas)
    key = cv2.waitKey(1) & 0xFF
    time.sleep(1)
    #
    #    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        # reset_lights()
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
