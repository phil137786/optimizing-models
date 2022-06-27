from tensorflow.keras.applications.resnet50 import ResNet50
import tf2onnx
import onnx
import tensorflow as tf


# load keras model

model = ResNet50(include_top=True, weights='imagenet')
input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)]

# convert to onnx model
onnx_model = tf2onnx.convert.from_keras(model, input_signature=input_signature)

onnx.save(onnx_model, "dst/path/model.onnx")