#not working

from tensorflow.python.compiler.tensorrt import trt_convert_windows as trt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

model1 = 'resnet50_saved_model'
model2 = 'mobileNetV2_saved_model'

output_saved_model_dir=''

def test1():
    # klappt nicht weil funktion nich da
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=model1)
    converter.convert()
    converter.save(output_saved_model_dir)

def test2():
    # funktioniert nicht weil Windows
    params = tf.experimental.tensorrt.ConversionParams(precision_mode='FP16')
    converter = tf.compiler.tensorrt.trt_convert_windows (input_saved_model_dir="resnet50_saved_model", conversion_params=params)
    converter.convert()
    converter.save('next test')

def ResNet50_einlesen():
    model = ResNet50(weights='imagenet')
    model.save('resnet50_saved_model') 

def MobileNetV2_einlesen():
    model = model = MobileNetV2(weights='imagenet')
    model.save('mobileNetV2_saved_model') 


