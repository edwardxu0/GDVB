import argparse
from pyfiglet import Figlet


def parse_args():
    f = Figlet(font="slant")
    print(f.renderText("GDVB"), end="")

    parser = argparse.ArgumentParser(
        description="Generative Diverse Verification Benchmarks for Nueral Network Verification",
        prog="GDVB",
    )

    parser.add_argument("configs", type=str, help="Configurations file.")
    parser.add_argument(
        "task",
        type=str,
        choices=["C", "T", "P", "V", "A", "E"],
        help="Select tasks to perform, including, compute [C]A, [T]rtain benchmark networks, generate [P]roperties, [V]erify benchmark instances, [A]nalyze benchmark results, and [E]verything above.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--result_dir", type=str, default="./results/", help="Result directory."
    )

    parser.add_argument(
        "--override", action="store_true", help="Override existing logs."
    )
    parser.add_argument("--debug", action="store_true", help="Print debug log.")
    parser.add_argument("--dumb", action="store_true", help="Silent mode.")
    parser.add_argument("--version", action="version", version="%(prog)s 2.0.0")

    return parser.parse_args()
