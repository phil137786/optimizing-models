import tensorflow as tf

TFLITE_MODEL='mobilenet_v2_1.0_224_quant.tflite'

interpreter = tf.lite.Interpreter(model_content=TFLITE_MODEL)
interpreter.allocate_tensors()  # Needed before execution!

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.
input_data = tf.constant(1., shape=[1, 1])
interpreter.set_tensor(input['index'], input_data)
interpreter.invoke()
interpreter.get_tensor(output['index']).shape
