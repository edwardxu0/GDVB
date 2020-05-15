"""
"""
import argparse

from .. import distillation
from .. import info

from . import utils as cli_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="r4v - refactoring for verification",
        prog="r4v",
        formatter_class=cli_utils.HelpFormatter,
        parents=[cli_utils.common_parser()],
    )
    subparsers = parser.add_subparsers(metavar="subcommand")

    # add subparsers here
    distillation.add_subparser(subparsers)
    info.add_subparser(subparsers)

    parser_help = subparsers.add_parser(
        "help",
        description="Displays the help message for a specified command",
        help="show a help message for a specified command",
        formatter_class=cli_utils.HelpFormatter,
    )
    subparser_dict = dict(subparsers.choices)
    parser_help.add_argument(
        "command",
        nargs="?",
        default="",
        choices=subparser_dict.keys(),
        help="the command to show the help of",
    )
    subparser_dict[""] = parser
    parser_help.set_defaults(
        func=lambda args: subparser_dict[args.command].print_help()
    )

    parser.set_defaults(func=lambda args: parser.print_help())

    return parser.parse_args()
