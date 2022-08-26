import tensorflow as tf
import tf2onnx
from tensorflow.keras.applications.resnet50 import ResNet50
import bentoml
model = ResNet50(weights='imagenet')
signatures = {
    "run": {"batchable": True},
}
bentoml.onnx.save_model("onnx_resnet50", onnx_model, signatures=signatures)
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)