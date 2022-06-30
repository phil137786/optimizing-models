#not woring

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import keras2onnx

model = MobileNetV2 (weights='imagenet')
output_model_path = "keras_efficientNet.onnx"
onnx_model = keras2onnx.convert_keras(model, model.name)
keras2onnx.save_model(onnx_model, output_model_path)