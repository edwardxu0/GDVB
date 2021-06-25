import numpy as np
import toml
import onnx
import onnxruntime
import torch

from pathlib import Path

from .artifact import Artifact

SIZE = int(1e5)
DATA_PATH = './data/acas'


class ACAS(Artifact):
    def __init__(self, dnn_configs):
        super().__init__(dnn_configs)
        _generate_acas_data(dnn_configs)

    def generate_property(self, prop_id):
        pass


def _data_generator(model, size):
    np.random.seed(0)
    x_min = np.array([0.0, -3.141593, -3.141593, 100.0, 0.0, ])
    x_max = np.array([60760.0, 3.141593, 3.141593, 1200.0, 1200.0, ])
    x_mean = np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0, ])
    x_range = np.array([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0, ])

    x_min_norm = (x_min - x_mean) / x_range
    x_max_norm = (x_max - x_mean) / x_range

    data = []
    for i in range(len(x_min)):
        data += [np.random.uniform(low=x_min_norm[i], high=x_max_norm[i], size=size)]
    data = np.array(data).T

    # run inference of the ACAS onnx models over the synthesized data
    # https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    onnx_model = onnx.load(model)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(model)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    label = []
    for x in data:
        x = torch.Tensor([x])
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)
        label += ort_outs
    label = np.array(label).reshape(size, 5)
    return data, label


def _generate_acas_data(dnn_configs):
    r4v_configs = toml.load(open(dnn_configs['r4v_config'], 'r'))
    train_path = Path(r4v_configs['distillation']['data']['train']['student']['path'])
    model = r4v_configs['distillation']['teacher']['model']

    train_path.mkdir(exist_ok=True, parents=True)
    train_data, train_label = _data_generator(model, SIZE)
    np.save(train_path.joinpath('data'), train_data)
    np.save(train_path.joinpath('label'), train_label)

    valid_path = train_path.parent.joinpath('valid')
    valid_path.mkdir(exist_ok=True, parents=True)
    valid_data, valid_label = _data_generator(model, int(SIZE/100))
    np.save(valid_path.joinpath('data'), valid_data)
    np.save(valid_path.joinpath('label'), valid_label)
