from tensorflow.python.compiler.tensorrt import trt_convert_windows as trt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50

# Es wird das MobileNetV2 Modell mit Standartgewichtung verwendet
model = MobileNetV2(weights='imagenet')

output_saved_model_dir=''

def test1():
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=model)
    converter.convert()
    converter.save(output_saved_model_dir)

def test2():
    params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP16')
    converter = tf.compiler.tensorrt.trt_convert_windows (input_saved_model_dir="resnet50_saved_model", conversion_params=params)
    converter.convert()
    converter.save('next test')

def modell_einlesen():
    model = ResNet50(weights='imagenet')
    model.save('resnet50_saved_model') 

test2()