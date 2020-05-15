#!/usr/bin/env python
import argparse
import numpy as np
import os
import shutil
import toml
import torch

import r4v.logging as logging

from r4v.distillation.config import DataConfiguration
from r4v.distillation.data import get_data_loader


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate properties for the DroNet network."
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
        "-g", "--gamma", type=float, default=10, help="The output radius to use."
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
                "id,property_filename,image_filename,numpy_filename,steering_angle,collision_prob,steering_angle_lb,steering_angle_ub,collision_prob_lb,collision_prob_ub\n"
            )

    config = DataConfiguration(toml.load(args.data_config))
    config.config["_STAGE"] = "val_test"
    config.config["shuffle"] = True
    config.config["batchsize"] = 1
    data_loader = get_data_loader(config)

    logger.info("Generating properties.")
    steer_count = 0
    coll_count = 0
    for idx, _, sx, target in data_loader:
        if steer_count >= args.num_properties and coll_count >= args.num_properties:
            break
        input_img_path = data_loader.dataset.samples[0][idx][0]
        new_img_path = os.path.join(
            args.output_dir, "%s%s" % (idx.item(), os.path.splitext(input_img_path)[-1])
        )

        property_type = None

        steering_angle = target[:, 0].item()
        steering_angle_lb, steering_angle_ub = float("nan"), float("nan")
        if not np.isnan(steering_angle) and steer_count < args.num_properties:
            steering_angle_lb = max(
                -np.pi / 2, steering_angle - args.gamma * np.pi / 180
            )
            steering_angle_ub = min(
                np.pi / 2, steering_angle + args.gamma * np.pi / 180
            )
            steer_count += 1
            property_type = "steer"

        collision_prob = target[:, 1].item()
        collision_prob_lb, collision_prob_ub = float("nan"), float("nan")
        if not np.isnan(collision_prob) and coll_count < args.num_properties:
            collision_prob_lb = 0.5 if collision_prob >= 0.5 else 0.0
            collision_prob_ub = 0.5 if collision_prob < 0.5 else 1.0
            coll_count += 1
            property_type = "collision"

        if property_type is None:
            continue

        shutil.copy(input_img_path, new_img_path)

        npy_img_path = os.path.join(args.output_dir, "%s.npy" % idx.item())
        img = sx[0].numpy()
        np.save(npy_img_path, img)

        property_path = os.path.join(args.output_dir, "robustness.%s.py" % idx.item())
        with open(property_path, "w+") as property_file:
            if property_type == "steer":
                property_file.write(
                    "from dnnv.properties import *\n"
                    "import numpy as np\n\n"
                    'N = Network("N")\n'
                    f'x = Image("{npy_img_path}")\n'
                    "input_layer = 0\n"
                    "output_layer = -1\n"
                    "output_select = 0\n\n"
                    f"epsilon = {args.epsilon}\n"
                    f"gamma = {args.gamma} * np.pi / 180\n"
                    "output = N[input_layer:output_layer, output_select](x)\n"
                    "gamma_lb = max(-np.pi / 2, (output - gamma) / 2)\n"
                    "gamma_ub = min(np.pi / 2, (output + gamma) / 2)\n"
                    "Forall(\n"
                    "    x_,\n"
                    "    Implies(\n"
                    "        ((x - epsilon) < x_ < (x + epsilon)),\n"
                    "        (gamma_lb < N[input_layer:output_layer, output_select](x_) < gamma_ub),\n"
                    "    ),\n"
                    ")\n"
                )
            elif property_type == "collision":
                property_file.write(
                    "from dnnv.properties import *\n"
                    'N = Network("N")\n'
                    f'x = Image("{npy_img_path}")\n'
                    "input_layer = 0\n"
                    "output_layer = -2\n"
                    "output_select = 1\n\n"
                    f"epsilon = {args.epsilon}\n"
                    f"gamma_lb = {collision_prob_lb}\n"
                    f"gamma_ub = {collision_prob_ub}\n"
                    "output = N[input_layer:output_layer, output_select](x)\n"
                    "Forall(\n"
                    "    x_,\n"
                    "    Implies(\n"
                    "        ((x - epsilon) < x_ < (x + epsilon)),\n"
                    "        Implies(output < 0.5, N[input_layer:output_layer, output_select](x_) < 0.0)\n"
                    "        & Implies(output > 0.5, N[input_layer:output_layer, output_select](x_) > 0.0),\n"
                    "    ),\n"
                    ")\n"
                )

        with open(properties_filename, "a") as prop_file:
            prop_file.write(
                "%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
                % (
                    idx.item(),
                    new_img_path,
                    npy_img_path,
                    steering_angle,
                    collision_prob,
                    steering_angle_lb,
                    steering_angle_ub,
                    collision_prob_lb,
                    collision_prob_ub,
                )
            )
    logger.info("Generated %d steering angle properties.", steer_count)
    logger.info("Generated %d collision probability properties.", coll_count)


if __name__ == "__main__":
    main(_parse_args())
