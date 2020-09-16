#!/usr/bin/env python
import numpy as np
from pathlib import Path

dave_orig_property_template = (
    "from dnnv.properties import *\n"
    "import numpy as np\n\n"
    'N = Network("N")\n'
    "means = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((1, 3, 1, 1))\n"
    "mins = np.zeros((1, 3, 100, 100)) - np.tile(means, (1, 1, 100, 100))\n"
    "maxs = np.zeros((1, 3, 100, 100)) + 255 - np.tile(means, (1, 1, 100, 100))\n"
    'x = Image("{image_path}") - np.tile(means, (1, 1, 100, 100))\n'
    "input_layer = 0\n"
    "output_layer = -2\n\n"
    'epsilon = Parameter("epsilon", type=float)\n'
    'gamma = Parameter("gamma", type=float, default=15.0) * np.pi / 180\n'
    "output = N[input_layer:](x)\n"
    "gamma_lb = np.tan(max(-np.pi / 2, (output - gamma) / 2))\n"
    "gamma_ub = np.tan(min(np.pi / 2, (output + gamma) / 2))\n"
    "Forall(\n"
    "    x_,\n"
    "    Implies(\n"
    "        (mins <= x_ <= maxs) & ((x - epsilon) < x_ < (x + epsilon)),\n"
    "        (gamma_lb < N[input_layer:output_layer](x_) < gamma_ub),\n"
    "    ),\n"
    ")\n"
)

dave_small_property_template = (
    "from dnnv.properties import *\n"
    "import numpy as np\n\n"
    'N = Network("N")\n'
    'x = Image("{image_path}") / 255.0\n'
    "input_layer = 0\n"
    "output_layer = -2\n\n"
    'epsilon = Parameter("epsilon", type=float) / 255.0\n'
    'gamma = Parameter("gamma", type=float, default=15.0) * np.pi / 180\n'
    "output = N[input_layer:](x)\n"
    "gamma_lb = np.tan(max(-np.pi / 2, (output - gamma) / 2))\n"
    "gamma_ub = np.tan(min(np.pi / 2, (output + gamma) / 2))\n"
    "Forall(\n"
    "    x_,\n"
    "    Implies(\n"
    "        (0 <= x_ <= 1) & ((x - epsilon) < x_ < (x + epsilon)),\n"
    "        (gamma_lb < N[input_layer:output_layer](x_) < gamma_ub),\n"
    "    ),\n"
    ")\n"
)


def main():
    properties_path = Path("properties")
    properties_path.mkdir(exist_ok=True)

    properties_filename = "properties.csv"
    with open(properties_filename, "w+") as prop_file:
        prop_file.write(
            "problem_id,property_filename,network_names,network_filenames\n"
        )

    networks = list(Path("onnx").iterdir())
    images = list(Path("original/images").iterdir())

    for i, image_path in enumerate(images):
        with open(image_path) as image_file:
            image = (
                np.array([float(v) for v in image_file.readline().split(",")[:-1]])
                .reshape((1, 3, 100, 100))
                .transpose(0, 1, 3, 2)[0]
                .astype(np.float32)
            )

        dave_orig_image_path = properties_path / f"dave_orig_image{i}.npy"
        np.save(dave_orig_image_path, image[[2, 1, 0]])

        dave_small_image_path = properties_path / f"dave_small_image{i}.npy"
        np.save(dave_small_image_path, image)

        dave_orig_property_path = properties_path / f"dave_orig_property_{i}.py"
        with open(dave_orig_property_path, "w+") as prop:
            prop.write(
                dave_orig_property_template.format(image_path=f"{dave_orig_image_path}")
            )

        dave_small_property_path = properties_path / f"dave_small_property_{i}.py"
        with open(dave_small_property_path, "w+") as prop:
            prop.write(
                dave_small_property_template.format(
                    image_path=f"{dave_small_image_path}"
                )
            )

        with open(properties_filename, "a") as prop_file:
            prop_file.write(
                f"DAVE_ORIG_{i},{dave_orig_property_path},N,onnx/dave.onnx\n"
            )
            prop_file.write(
                f"DAVE_SMALL_{i},{dave_small_property_path},onnx/dave_small.onnx\n"
            )

    # config = DataConfiguration(toml.load(args.data_config))
    # config.config["_STAGE"] = "val_test"
    # config.config["shuffle"] = True
    # config.config["batchsize"] = 1
    # data_loader = get_data_loader(config)

    # logger.info("Generating properties.")
    # for i, (idx, _, sx, target) in enumerate(data_loader):
    #     if i == args.num_properties:
    #         break
    #     input_img_path = data_loader.dataset.samples[0][idx][0]
    #     new_img_path = os.path.join(
    #         args.output_dir, "%s%s" % (idx.item(), os.path.splitext(input_img_path)[-1])
    #     )
    #     shutil.copy(input_img_path, new_img_path)

    #     npy_img_path = os.path.join(args.output_dir, "%s.npy" % idx.item())

    #     img = sx[0].numpy()
    #     np.save(npy_img_path, img)

    #     steering_angle = target.item()
    #     steering_angle_lb = max(-np.pi / 2, steering_angle - args.gamma * np.pi / 180)
    #     steering_angle_ub = min(np.pi / 2, steering_angle + args.gamma * np.pi / 180)

    #     property_path = os.path.join(args.output_dir, "robustness.%s.py" % idx.item())
    #     with open(property_path, "w+") as property_file:
    #         property_file.write(
    #             "from dnnv.properties import *\n"
    #             "import numpy as np\n\n"
    #             'N = Network("N")\n'
    #             f'x = Image("{npy_img_path}")\n'
    #             "input_layer = 0\n"
    #             "output_layer = -2\n\n"
    #             f"epsilon = {args.epsilon}\n"
    #             f"gamma = {args.gamma} * np.pi / 180\n"
    #             "output = N[input_layer:](x)\n"
    #             "gamma_lb = np.tan(max(-np.pi / 2, (output - gamma) / 2))\n"
    #             "gamma_ub = np.tan(min(np.pi / 2, (output + gamma) / 2))\n"
    #             "Forall(\n"
    #             "    x_,\n"
    #             "    Implies(\n"
    #             "        ((x - epsilon) < x_ < (x + epsilon)),\n"
    #             "        (gamma_lb < N[input_layer:output_layer](x_) < gamma_ub),\n"
    #             "    ),\n"
    #             ")\n"
    #         )

    #     with open(properties_filename, "a") as prop_file:
    #         prop_file.write(
    #             "%s,%s,%s,%s,%s,%s,%s\n"
    #             % (
    #                 idx.item(),
    #                 property_path,
    #                 new_img_path,
    #                 npy_img_path,
    #                 steering_angle,
    #                 steering_angle_lb,
    #                 steering_angle_ub,
    #             )
    #         )
    # logger.info("Generated %d properties.", i)


if __name__ == "__main__":
    main()
