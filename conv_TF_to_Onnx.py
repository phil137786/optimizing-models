#not working

from importlib import import_module
from tensorflow.keras.applications.resnet50 import ResNet50
import tf2onnx
import tensorflow as tf
import numpy as np
import onnx


BATCH_SIZE = 32
shape = (BATCH_SIZE, 224, 224, 3)

input_signature = [tf.TensorSpec([shape], tf.float32, name='x')]

# load keras model
model = ResNet50(include_top=True, weights='imagenet')


output_model_path = "keras_efficientNet.onnx"
# convert model to ONNX
onnx_model = tf2onnx.convert.from_keras(model, input_signature=input_signature)

onnx.save_model(onnx_model, "example.onnx")