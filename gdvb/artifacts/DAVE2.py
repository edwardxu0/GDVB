import os
import shutil
import numpy as np
import torch

from .artifact import Artifact

from r4v.distillation.config import DataConfiguration
from r4v.distillation.data import get_data_loader


class DAVE2(Artifact):
    def __init__(self, dnn_configs):
        super().__init__(dnn_configs)
        self.__name__ = "DAVE2"
        self.input_shape = (3, 100, 100)

    def generate_property(
        self, data_config, prop_id, epsilon, skip_layers, output_dir, seed, gamma=15
    ):
        torch.manual_seed(seed)
        config = DataConfiguration(data_config)
        config.config["_STAGE"] = "val_test"
        config.config["shuffle"] = True
        config.config["batchsize"] = 1

        config.config["teacher"] = {}
        config.config["teacher"]["path"] = config.config["test"]["teacher"]["path"]
        config.config["student"] = {}
        config.config["student"]["path"] = config.config["test"]["student"]["path"]
        data_loader = get_data_loader(config)

        for i, (idx, _, sx, target) in enumerate(data_loader):
            if i == prop_id:
                input_img_path = data_loader.dataset.samples[0][idx][0]
                new_img_path = os.path.join(
                    output_dir, f"{idx.item()}{os.path.splitext(input_img_path)[-1]}"
                )
                shutil.copyfile(input_img_path, new_img_path)

                npy_img_path = os.path.join(output_dir, f"{idx.item()}.npy")

                img = sx[0].numpy()
                np.save(npy_img_path, img)

                property_lines = [
                    "from dnnv.properties import *\n",
                    "import numpy as np\n\n",
                    'N = Network("N")\n',
                    f'x = Image("{npy_img_path}")\n',
                ]

                if not skip_layers or skip_layers == 0:
                    property_lines += [f"input_layer = 0\n"]
                else:
                    property_lines += [f"input_layer = {skip_layers}\n"]

                property_lines += [
                    "output_layer = -2\n\n",
                    f"epsilon = {epsilon}\n",
                    f"gamma = {gamma} * np.pi / 180\n",
                    "output = N[input_layer:](x)\n",
                    "gamma_lb = np.tan(max(-np.pi / 2, (output - gamma) / 2))\n",
                    "gamma_ub = np.tan(min(np.pi / 2, (output + gamma) / 2))\n",
                    "Forall(\n",
                    "    x_,\n",
                    "    Implies(\n",
                    "        ((x - epsilon) < x_ < (x + epsilon)),\n",
                    "        (gamma_lb < N[input_layer:output_layer](x_) < gamma_ub),\n",
                    "    ),\n",
                    ")\n",
                ]

                property_path = os.path.join(output_dir, f"robustness_{i}_{epsilon}.py")
                with open(property_path, "w+") as property_file:
                    property_file.writelines(property_lines)
                print(i, target, prop_id)
                break
