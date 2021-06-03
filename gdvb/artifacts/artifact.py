from ..nn.onnxu import ONNXU

from ..nn.layers import Dense, Conv, Transpose, Flatten


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

    # get the layer sizes that are used in layer scaling factor calculation
    # FC: number of neurons
    # Conv: number of kernels
    # Others: -1
    def get_layer_sizes(self):
        layer_sizes = []
        for layer in self.layers:
            if isinstance(layer, (Dense, Conv)):
                layer_sizes += [layer.size]
            elif isinstance(layer, (Flatten, Transpose)):
                layer_sizes += [-1]
            else:
                raise NotImplementedError(f'Unsupported layer type: {layer}')
        return layer_sizes