import numpy as np
import os
import pathlib
import toml

from .artifact import Artifact

SIZE = 1e5
DATA_PATH = './data/acas'


class ACAS(Artifact):
    def __init__(self, dnn_configs):
        super().__init__(dnn_configs)
        self._generate_acas_data(dnn_configs['data_config'])

    def _generate_acas_data(self, data_config_path):

        data_config = toml.load(open(data_config_path, 'r'))
        assert data_config['student']['path'] == data_config['teacher']['path']
        data_path = data_config['student']['path']

        train_path = os.path.join(data_path, 'train')
        pathlib.Path.mkdir(train_path, exist_ok=True, parents=True)
        train_data = self._data_generator(SIZE)
        np.save(train_path, train_data)




    def _data_generator(self, size):
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
        return data