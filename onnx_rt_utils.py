import logging as logger
import numpy as np
import os
import onnxruntime as ort
 
 
class OnnxRuntimeModel:
    def __init__(self, model_path, device="cpu"):
        self.model = None
 
        providers = ["CPUExecutionProvider"]
        if device == "gpu":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
 
        if model_path:
            self.load_model(model_path, providers)
 
    def __call__(self, **kwargs):
        inputs = {k: np.array(v) for k, v in kwargs.items()}
        return self.model.run(None, inputs)
 
    def load_model(self, path: str, providers=None, sess_options=None):
        if providers is None:
            logger.info("No onnxruntime provider specified, using CPUExecutionProvider")
            providers = ["CPUExecutionProvider"]  # "CUDAExecutionProvider",
 
        self.model = ort.InferenceSession(path, providers=providers, sess_options=sess_options)
 
 
def get_random_data(shape, dtype, args=None):
    min_value = -1
    max_value = 1
 
    if dtype.find("int") >= 0:
        min_value = 0
    data = np.random.uniform(min_value, max_value, size=shape).astype(dtype)
    # data = np.minimum(data, 3)
    return data