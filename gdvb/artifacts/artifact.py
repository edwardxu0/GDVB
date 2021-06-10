from ..nn.onnxu import ONNXU


class Artifact:
    def __init__(self, dnn_configs):
        self.dnn_configs = dnn_configs
        self.onnx = ONNXU(dnn_configs['onnx'])
        self.layers = self._get_layers()

    def _get_layers(self):
        supported_layers = self.dnn_configs['supported_layers']
        start_layer = self.dnn_configs['start_layer']
        layers = []
        for layer in self.onnx.arch:
            if layer.type in supported_layers:
                layers += [layer]
        return layers[start_layer:]
