#source: https://colab.research.google.com/github/frogermcs/TFLite-Tester/blob/master/notebooks/Testing_TFLite_model.ipynb#scrollTo=jor83-LqI8xW


import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np

model = MobileNetV2(weights='imagenet')

folder_path = "prepared_img224x224"

from tensorflow.keras.preprocessing.image import load_img
data = np.empty((10, 224, 224, 3),dtype=np.float32)
for i in range(10):
    data[i] = load_img('prepared_img224x224/' + str(i) + '.jpg')

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
val_image_batch = preprocess_input(data)


# A generator that provides a representative dataset
def representative_data_gen():
    yield [val_image_batch]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open('mobilenet_v2_1.0_224_quant.tflite', 'wb') as f:
  f.write(tflite_model)