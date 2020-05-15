"""
"""
import argparse

from .. import dispatcher
from .. import distillation
from ..config import parse as parse_config


def dispatch(args: argparse.Namespace):
    config = parse_config(
        args.config,
        override={
            key: value
            for key, value in vars(args).items()
            if key.startswith("distillation")
        },
    )
    distillation_config = config.distillation
    dispatcher.dispatch(
        target=distillation.distill,
        args=(distillation_config,),
        max_memory=config.distillation.max_memory,
        timeout=config.distillation.timeout,
    )


def add_subparser(subparsers: argparse._SubParsersAction):
    from ..cli import utils as cli_utils

    parser = subparsers.add_parser(
        "distill",
        description="Distill a teacher model to a verifiable student model",
        help="distill a teacher model to a verifiable student model",
        formatter_class=cli_utils.HelpFormatter,
        parents=[cli_utils.common_parser()],
    )

    parser.add_argument("config", help="configuration for distillation")

    parser.add_argument(
        "--distillation.novalidation",
        "--novalidation",
        action="store_true",
        help="do not perform validation after each training epoch",
    )

    parser.add_argument(
        "-n",
        "--distillation.parameters.epochs",
        "--epochs",
        default=None,
        type=int,
        help="the maximum number of epochs",
    )
    parser.add_argument(
        "-lr",
        "--distillation.parameters.learning_rate",
        "--learning-rate",
        default=None,
        type=float,
        help="the learning rate to use for distillation",
    )
    parser.add_argument(
        "-wd",
        "--distillation.parameters.weight_decay",
        "--weight-decay",
        default=None,
        type=float,
        help="the weight decay value to use for distillation",
    )
    parser.add_argument(
        "-m",
        "--distillation.parameters.momentum",
        "--momentum",
        default=None,
        type=float,
        help="the momentum value to use for distillation",
    )
    parser.add_argument(
        "-T",
        "--distillation.parameters.T",
        "--temperature",
        default=None,
        type=float,
        help="the temperature to use for distillation",
    )
    parser.add_argument(
        "-a",
        "--distillation.parameters.alpha",
        "--alpha",
        default=None,
        type=float,
        help="the soft/hard mixture ratio to use for distillation",
    )

    parser.add_argument(
        "-b",
        "--distillation.data.batchsize",
        "--batchsize",
        default=None,
        type=float,
        help="the batch size to use for distillation",
    )

    parser.set_defaults(func=dispatch)
