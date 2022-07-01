import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

model = model = MobileNetV2(weights='imagenet')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('mobileNetV2_2.tflite', 'wb') as f:
  f.write(tflite_model)