import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

tflite_flag = 0
input_saved_model_dir = "ssd_mobilenet_v2_2"
output_saved_model_dir = "ssd_mobilenet_v2_2_trt"

if tflite_flag == 1:
  converter = tf.lite.TFLiteConverter.from_saved_model(input_saved_model_dir)
  tflite_model = converter.convert()

  with open('ssd_v2.tflite', 'wb') as f:
    f.write(tflite_model)

else:

  conversion_params = trt.TrtConversionParams(
    precision_mode=trt.TrtPrecisionMode.FP32)
  
  converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=input_saved_model_dir,
    conversion_params=conversion_params)

  converter.convert()
  converter.save(output_saved_model_dir)