#!/bin/bash
cd /home/pi/Hand-Gesture-Recognition
source /env/bin/activate
python raw_recognize.py --conf config/config.json
