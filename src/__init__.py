import onnxruntime as rt


class BaseInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.load_model()
        
    def load_model(self, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]):
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = rt.InferenceSession(self.model_path, sess_options, providers)
        self.input_names = [input.name for input in self.sess.get_inputs()]
        self.output_names = [output.name for output in self.sess.get_outputs()]
        self.input_size = self.sess.get_inputs()[0].shape[2:0:-1]