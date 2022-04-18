import os
import sys

import serial
import time


def arduino_serial_write(x, port):
    try:
        arduino = serial.Serial(port=port, baudrate=115200, timeout=.1)
        arduino.write(bytes(x, 'utf-8'))
        time.sleep(0.05)
        data = arduino.readline()
        return data

    except Exception as ex:
        print("Arduino Error: Device not Detected", ex)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        f_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, f_name, exc_tb.tb_lineno)
        print("Check the port or Restart the system and try again")

# while True:
#     num = input("Enter a number: ")
#     value = write_read(num)
#     print(value)
