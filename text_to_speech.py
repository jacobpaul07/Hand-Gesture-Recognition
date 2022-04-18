# Import the required module for text
# to speech conversion
import os
import sys
import pyttsx3

# This module is imported so that we can
# play the converted audio


def text_to_speech(text, enable):
    try:
        if enable == "Enable":
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()

    except Exception as ex:
        print("Text_To_Speech", ex)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        f_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, f_name, exc_tb.tb_lineno)

