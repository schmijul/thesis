import tensorflow as tf
import tf2onnx
import onnx


model = tf.keras.models.load_model('trainedModels/deepkriging/Synthetic_slice_from-401_to-801-8-random-1_notnormalized/best_model.h5')
print('loaded model')

onnx_model, _ = tf2onnx.convert.from_keras(model)
print('transformed model')
onnx.save(onnx_model, "trainedModels/deepkriging/Synthetic_slice_from-401_to-801-8-random-1_notnormalized/model.onnx")
print('saved onnx model')