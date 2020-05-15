"""
"""
import argparse

from .. import config
from .. import dispatcher
from .. import distillation
from .. import info


def add_subparser(subparsers: argparse._SubParsersAction):
    from ..cli import utils as cli_utils

    parser = subparsers.add_parser(
        "info",
        description="Display information about an onnx model.",
        help="display info about an onnx model",
        formatter_class=cli_utils.HelpFormatter,
        parents=[cli_utils.common_parser()],
    )

    parser.add_argument("model", help="path to onnx model")

    parser.add_argument("--input_shape", type=int, nargs="+", default=[1, 3, 224, 224])
    parser.add_argument("--input_format", type=str, default="NCHW")
    parser.add_argument(
        "--plugins", type=str, nargs="+", default=[], help="plugin modules to load"
    )

    parser.set_defaults(func=info.show)
