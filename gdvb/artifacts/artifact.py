from ..nn.onnxu import ONNXU


class Artifact:
    def __init__(self, dnn_configs):

        self.supported_layers = ["Conv", "FC", "Transpose", "Flatten"]
        self.start_layer = 0

        self.dnn_configs = dnn_configs
        self.onnx = ONNXU(dnn_configs["onnx"])
        self.layers = self._get_layers()

    def _get_layers(self):
        layers = []
        for layer in self.onnx.arch:
            if layer.type in self.supported_layers:
                layers += [layer]
        return layers[self.start_layer :]
