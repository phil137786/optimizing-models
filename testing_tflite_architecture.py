import tensorflow as tf
import cv2 
import pathlib
import sys
import numpy as np



TFLITE_MODEL='mobilenet_v2_1.0_224.tflite'
#TFLITE_MODEL='mobilenet_v2_1.0_224_quant.tflite'



tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])

print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])