import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np

model = MobileNetV2(weights='imagenet')

folder_path = "prepared_img224x224"

# Print images batch and labels predictions for TFLite Model
from tensorflow.keras.preprocessing.image import load_img
data = np.empty((1, 224, 224, 3),dtype=np.float32)
data[0] = load_img('prepared_img224x224/' + str(0) + '.jpg')

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
val_image_batch = preprocess_input(data)

# Das Ergebnis wird ermittelt
predictions = model.predict(val_image_batch)

output_neuron = np.argmax(predictions)
print(output_neuron)