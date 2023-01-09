import os
import numpy as np
import torch
import toml

from PIL import Image

from .artifact import Artifact

from r4v.distillation.config import DataConfiguration
from r4v.distillation.data import get_data_loader


class TaxiNet(Artifact):
    def __init__(self, dnn_configs):
        super().__init__(dnn_configs)
        r4v_config = toml.load(dnn_configs["r4v_config"])
        self.input_shape = tuple(
            r4v_config["distillation"]["teacher"]["input_shape"][1:]
        )

    def generate_property(
        self, data_config, prop_id, epsilon, skip_layers, output_dir, seed
    ):
        torch.manual_seed(seed)
        config = DataConfiguration(data_config)
        config.config["_STAGE"] = "validation"
        config.config["shuffle"] = True
        config.config["batchsize"] = 1
        data_loader = get_data_loader(config)
        student_data_config = config.config["transform"]["student"]

        for i, (idx, _, sx, target) in enumerate(data_loader):
            if i == prop_id:
                new_img_path = os.path.join(output_dir, f"{idx.item()}.png")
                height = student_data_config["height"]
                width = student_data_config["width"]
                img = Image.fromarray(sx.numpy().reshape(height, width), "L")
                img.save(new_img_path)

                npy_img_path = os.path.join(output_dir, f"{idx.item()}.npy")
                image_shape = (1, self.input_shape[0], height, width)
                np.save(npy_img_path, sx.reshape(1, self.input_shape[0], height, width))
                # print(sx.reshape(1, self.input_shape[0], height, width).shape)

                property_path = os.path.join(output_dir, f"robustness_{i}_{epsilon}.py")

                gamma = 0.5
                property_lines = [
                    "from dnnv.properties import *",
                    "import numpy as np",
                    "",
                    'N = Network("N")',
                    f'x = Image("{npy_img_path}")',
                    "",
                    f"epsilon = {epsilon}",
                    f"gamma = {gamma}",
                    "output = N(x)",
                    "lb = output - gamma",
                    "ub = output + gamma",
                    "",
                    "Forall(",
                    "    x_,",
                    "    Implies(",
                    "        ((x - epsilon) < x_ < (x + epsilon)),",
                    "        (lb < N(x_) < ub),",
                    "    ),",
                    ")",
                ]
                property_lines = [x + "\n" for x in property_lines]

                with open(property_path, "w+") as property_file:
                    property_file.writelines(property_lines)
                break
