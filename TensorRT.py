from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# Es wird das MobileNetV2 Modell mit Standartgewichtung verwendet
model = MobileNetV2(weights='imagenet')

input_saved_model_dir=''
output_saved_model_dir=''

converter = trt.TrtGraphConverterV2(input_saved_model_dir=model)
converter.convert()
converter.save(output_saved_model_dir)