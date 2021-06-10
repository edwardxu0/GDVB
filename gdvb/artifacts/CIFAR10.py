import os
import numpy as np
import torch

from PIL import Image

from .artifact import Artifact

from r4v.distillation.config import DataConfiguration
from r4v.distillation.data import get_data_loader


class CIFAR10(Artifact):
    def __init__(self, dnn_configs):
        super().__init__(dnn_configs)
        self.input_shape = (3, 32, 32)

    def generate_property(self, data_config, prop_id, epsilon, skip_layers, output_dir, seed):
        torch.manual_seed(seed)
        config = DataConfiguration(data_config)
        config.config["_STAGE"] = "val_test"
        config.config["shuffle"] = True
        config.config["batchsize"] = 1
        data_loader = get_data_loader(config)

        for i, (idx, _, sx, target) in enumerate(data_loader):
            if i == prop_id:
                new_img_path = os.path.join(output_dir, f"{idx.item()}.png")
                height = config.config['transform']['height']
                width = config.config['transform']['width']
                # img = Image.fromarray(sx.numpy().reshape(self.input_shape[0], height, width), 'L')
                # img.save(new_img_path)

                npy_img_path = os.path.join(output_dir, f"{idx.item()}.npy")
                image_shape = (self.input_shape[0], height, width)
                np.save(npy_img_path, sx.reshape(self.input_shape[0], height, width))

                property_path = os.path.join(output_dir, f"robustness_{i}_{epsilon}.py")

                property_lines = ["from dnnv.properties import *\n",
                                  "import numpy as np\n\n",
                                  'N = Network("N")\n']
                
                if not skip_layers or skip_layers == 0:
                    property_lines += [f'x = Image("{npy_img_path}")\n',
                                       f"input_layer = 0\n"]
                else:
                    property_lines += [f'x = Image("{npy_img_path}").reshape((1,{np.prod(image_shape)}))\n',
                                       f"input_layer = {skip_layers}\n"]
                
                property_lines += [f"epsilon = {epsilon}\n",
                                   "Forall(\n",
                                   "    x_,\n",
                                   "    Implies(\n",
                                   "        ((x - epsilon) < x_ < (x + epsilon)),\n",
                                   "        argmax(N[input_layer:](x_)) == argmax(N[input_layer:](x)),\n",
                                   "    ),\n",
                                   ")\n"]
                
                with open(property_path, "w+") as property_file:
                    property_file.writelines(property_lines)
                break
