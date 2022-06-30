import tensorflow as tf
import cv2 
import pathlib
import sys
import numpy as np

folder_path = "prepared_img224x224"
TFLITE_MODEL='tf_lite_model.tflite'

tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)

def information():
    # Load TFLite model and see some details about input/output
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

# Print images batch and labels predictions for TFLite Model
from tensorflow.keras.preprocessing.image import load_img
data = np.empty((1, 224, 224, 3),dtype=np.float32)
data[0] = load_img('prepared_img224x224/' + str(0) + '.jpg')

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
val_image_batch = preprocess_input(data)


tflite_interpreter.allocate_tensors()  # Needed before execution!

output = tflite_interpreter.get_output_details()[0]  # Model has single output.
input = tflite_interpreter.get_input_details()[0]  # Model has single input.

tflite_interpreter.set_tensor(input['index'], val_image_batch)
tflite_interpreter.invoke()
print(tflite_interpreter.get_tensor(output['index']).shape)

x=np.argmax(tflite_interpreter.get_tensor(output['index']))
print(x)
