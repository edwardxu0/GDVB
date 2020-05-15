#!/usr/bin/env python
import argparse
import onnx
import toml
import torch

from r4v.distillation.config import DataConfiguration
from r4v.distillation.data import get_data_loader
from r4v.nn import load_network


def _parse_args():
    parser = argparse.ArgumentParser(description="Measure the performance of a model.")

    parser.add_argument("model_path", type=str, help="path to model")
    parser.add_argument("data_config", type=str, help="path to the dataset")

    parser.add_argument("--input_shape", type=int, nargs="+", default=[1, 3, 224, 224])
    parser.add_argument("--input_format", type=str, default="NCHW")

    parser.add_argument(
        "-p",
        "--prediction_type",
        default="classification",
        choices=["classification", "regression"],
    )

    parser.add_argument("--cuda", action="store_true")

    return parser.parse_args()


def main(args):
    config = DataConfiguration(toml.load(args.data_config))
    config.config["_STAGE"] = "val_test"
    data_loader = get_data_loader(config)

    model = load_network(
        {
            "model": args.model_path,
            "input_shape": args.input_shape,
            "input_format": args.input_format,
        }
    ).as_pytorch(maintain_weights=True)

    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    print("Using device: %s" % device)
    model.to(device)

    model.eval()

    prediction_type = args.prediction_type
    num_correct = 0.0
    num_samples = 0.0
    loss = 0.0
    for i, (idx, _, sx, target) in enumerate(data_loader):
        y = model(sx.to(device)).squeeze()
        target = target.to(device)
        num_samples += target.shape[0]
        if prediction_type == "classification":
            pred = y.argmax(-1)
            correct = pred == target
            num_correct += correct.sum(dtype=torch.float)
            performance = {"accuracy": (num_correct / num_samples).item()}
        else:
            loss += ((y - target) ** 2).sum()
            performance = {"loss": (loss / num_samples).item()}
        print("%7d: %s" % ((i + 1) * data_loader.batch_size, performance))


if __name__ == "__main__":
    main(_parse_args())
