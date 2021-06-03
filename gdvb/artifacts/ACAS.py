from .artifact import Artifact


class ACAS(Artifact):
    def __init__(self, dnn_configs):
        super().__init__(dnn_configs)
