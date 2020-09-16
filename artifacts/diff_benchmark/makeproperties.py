#!/usr/bin/env python
import numpy as np
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from dnnv.nn import parse
from pathlib import Path


def main():
    properties_path = Path("properties")
    properties_path.mkdir(exist_ok=True)

    global_properties_filename = "global_properties.csv"
    local_properties_filename = "local_properties.csv"

    with open(global_properties_filename, "w+") as prop_file:
        prop_file.write(
            "problem_id,property_filename,network_names,network_filenames\n"
        )
    with open(local_properties_filename, "w+") as prop_file:
        prop_file.write(
            "problem_id,property_filename,network_names,network_filenames\n"
        )
    with open("global_equivalence_template.txt") as template:
        global_equivalence_template = template.read()
    with open("global_targeted_diff_template.txt") as template:
        global_targeted_diff_template = template.read()
    with open("local_equivalence_template.txt") as template:
        local_equivalence_template = template.read()
    with open("local_targeted_diff_template.txt") as template:
        local_targeted_diff_template = template.read()

    networks = list(Path("onnx").iterdir())
    models = [parse(network) for network in networks]

    cifar10_test = datasets.CIFAR10(
        "/tmp/data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    cifar10_test_loader = data.DataLoader(cifar10_test, batch_size=1, shuffle=False)

    local_inputs = []
    for data_idx, (x, y) in enumerate(cifar10_test_loader):
        if len(local_inputs) == 10:
            break
        true_class = y.item()
        image = x.numpy()[0]

        predicted_classes = set()
        for m in models:
            predicted_classes.add(np.argmax(m(image[None, :])).item())
        if len(predicted_classes) > 1 or predicted_classes.pop() != true_class:
            continue
        local_idx = len(local_inputs)
        print(f"Using data point {data_idx} for local input {local_idx}")

        image_path = properties_path / f"input_{local_idx}.npy"
        np.save(image_path, image)
        local_inputs.append((image_path, true_class))

    for n1_idx, n1 in enumerate(networks):
        for n2 in networks[n1_idx + 1 :]:
            property_path = properties_path / f"global_equivalence_property.py"
            with open(property_path, "w+") as prop:
                prop.write(global_equivalence_template)
            with open(global_properties_filename, "a") as prop_file:
                prop_file.write(f"GLOBAL_EQ,{property_path},N1:N2,{n1}:{n2}\n")

            for c1 in range(10):
                for c2 in range(10):
                    if c1 == c2:
                        continue
                    property_path = (
                        properties_path / f"global_targeted_diff_{c1}_{c2}.py"
                    )
                    with open(property_path, "w+") as prop:
                        prop.write(
                            global_targeted_diff_template.format(class_1=c1, class_2=c2)
                        )
                    with open(global_properties_filename, "a") as prop_file:
                        prop_file.write(
                            f"GLOBAL_TDIFF_{c1}_{c2},{property_path},N1:N2,{n1}:{n2}\n"
                        )

            for local_idx, (image_path, true_class) in enumerate(local_inputs):
                property_path = (
                    properties_path / f"local_equivalence_property_{local_idx}.py"
                )
                with open(property_path, "w+") as prop:
                    prop.write(
                        local_equivalence_template.format(input_path=f"{image_path}")
                    )
                with open(local_properties_filename, "a") as prop_file:
                    prop_file.write(
                        f"LOCAL_EQ_{local_idx},{property_path},N1:N2,{n1}:{n2}\n"
                    )

                for other_class in range(10):
                    if other_class == true_class:
                        continue
                    property_path = (
                        properties_path
                        / f"local_targeted_diff_{true_class}_{other_class}_property_{local_idx}.py"
                    )
                    with open(property_path, "w+") as prop:
                        prop.write(
                            local_targeted_diff_template.format(
                                input_path=f"{image_path}",
                                true_class=true_class,
                                other_class=other_class,
                            )
                        )
                    with open(local_properties_filename, "a") as prop_file:
                        prop_file.write(
                            f"LOCAL_DIFF_{true_class}_{other_class}_{local_idx},{property_path},N1:N2,{n1}:{n2}\n"
                        )


if __name__ == "__main__":
    main()
