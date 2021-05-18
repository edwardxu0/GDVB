#!/usr/bin/env python
import argparse
import numpy as np
import os
import shutil
import toml
import torch

from PIL import Image

import r4v.logging as logging

from r4v.distillation.config import DataConfiguration
from r4v.distillation.data import get_data_loader


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate properties for the mnist networks."
    )

    parser.add_argument("data_config", type=str, help="data configuration file")
    parser.add_argument("output_dir", type=str, help="output directory")

    parser.add_argument(
        "-N",
        "--num_properties",
        type=int,
        default=10,
        help="the number of properties to generate",
    )
    parser.add_argument(
        "-e", "--epsilon", type=float, default=2, help="The input radius to use."
    )
    parser.add_argument(
        "-i", "--input_dimension", type=int, default=-1, help="input_dimension"
    )
    
    parser.add_argument(
        "--rm_transpose", action='store_true', help="remove the first three layers"
    )
    
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.set_defaults(debug=True)
    return parser.parse_args()


def main(args):
    logger = logging.initialize(__name__, args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    properties_filename = os.path.join(args.output_dir, "properties.csv")
    if not os.path.exists(properties_filename):
        with open(properties_filename, "w+") as prop_file:
            prop_file.write(
                "id,property_filename,image_filename,numpy_filename,target\n"
            )

    config = DataConfiguration(toml.load(args.data_config))
    config.config["_STAGE"] = "val_test"
    config.config["shuffle"] = True
    config.config["batchsize"] = 1
    data_loader = get_data_loader(config)

    logger.info("Generating properties.")
    for i, (idx, _, sx, target) in enumerate(data_loader):
        if i == args.num_properties:
            break
        
        new_img_path = os.path.join(
            args.output_dir, "%s.png" % (idx.item())
        )
        height = config.config['transform']['height']
        width = config.config['transform']['width']
        assert height == width
        img = Image.fromarray(sx.numpy().reshape(height,width), 'L')
        img.save(new_img_path)
        
        npy_img_path = os.path.join(args.output_dir, "%s.npy" % idx.item())
        np.save(npy_img_path, sx.reshape(1,height,width))

        property_path = os.path.join(args.output_dir, "robustness.%s.%s.py" % (idx.item(), args.epsilon))
        if args.rm_transpose:
            with open(property_path, "w+") as property_file:
                property_file.write(
                    "from dnnv.properties import *\n"
                    "import numpy as np\n\n"
                    'N = Network("N")\n'
                    f'x = Image("{npy_img_path}").reshape((1,{args.input_dimension}))\n' # REMOVE RESHAPE
                    "input_layer = 2\n" # START FROM 2(my mnist)/3(eran mnist)
                    f"epsilon = {args.epsilon}\n"
                    "Forall(\n"
                    "    x_,\n"
                    "    Implies(\n"
                    "        ((x - epsilon) < x_ < (x + epsilon)),\n"
                    "        argmax(N[input_layer:](x_)) == argmax(N[input_layer:](x)),\n"
                    "    ),\n"
                    ")\n"
                )
        else:
            with open(property_path, "w+") as property_file:
                property_file.write(
                    "from dnnv.properties import *\n"
                    "import numpy as np\n\n"
                'N = Network("N")\n'
                f'x = Image("{npy_img_path}")\n'# .reshape((1,{args.input_dimension}))\n' # REMOVE RESHAPE
                "input_layer = 0\n" # START FROM 0
                f"epsilon = {args.epsilon}\n"
                "Forall(\n"
                "    x_,\n"
                "    Implies(\n"
                "        ((x - epsilon) < x_ < (x + epsilon)),\n"
                "        argmax(N[input_layer:](x_)) == argmax(N[input_layer:](x)),\n"
                "    ),\n"
                ")\n"
            )

        with open(properties_filename, "a") as prop_file:
            prop_file.write(
                "%s,%s,%s,%s,%s\n"
                % (
                    idx.item(),
                    property_path,
                    new_img_path,
                    npy_img_path,
                    target
                )
            )
    logger.info("Generated %d properties.", i)


if __name__ == "__main__":
    main(_parse_args())
