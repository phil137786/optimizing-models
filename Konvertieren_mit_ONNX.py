from tensorflow.keras.applications.resnet50 import ResNet50
import keras2onnx
import tensorflow as tf
import numpy as np


BATCH_SIZE = 32
shape = (BATCH_SIZE, 224, 224, 3)

# load keras model
model = ResNet50(include_top=True, weights='imagenet')


#output_model_path = "keras_efficientNet.onnx"
#onnx_model = keras2onnx.convert_keras(model, model.name)
#keras2onnx.save_model(onnx_model, output_model_path)