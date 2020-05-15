#!/usr/bin/env python
import argparse
import importlib.util
import numpy as np
import onnxmltools  # uses onnxmltools==1.3
import onnx
import os
import sys

from keras.models import load_model
from onnx import helper, numpy_helper


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the dave network to an onnx model."
    )

    parser.add_argument(
        "dave_path",
        type=str,
        help="path to file defining dave model (keras model accessible by call to function DAVE)",
    )
    parser.add_argument(
        "output_path", type=str, help="where to write the resulting onnx model"
    )
    return parser.parse_args()


def main(args):
    spec = importlib.util.spec_from_file_location("model", args.dave_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    onnx_model = onnxmltools.convert_keras(model.DAVE())
    onnxmltools.utils.save_model(onnx_model, args.output_path)


if __name__ == "__main__":
    main(_parse_args())
