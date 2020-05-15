"""
"""
import importlib

from pathlib import Path
from .. import logging
from .cli import add_subparser
from ..nn import load_network, Droppable, Linearizable, Rescalable


class Color:
    NONE = "\033[0m"
    BLACK = "\033[1;30;40m"
    RED = "\033[1;31;40m"
    GREEN = "\033[1;32;40m"
    YELLOW = "\033[1;33;40m"
    BLUE = "\033[1;34;40m"
    PURPLE = "\033[1;35;40m"
    CYAN = "\033[1;36;40m"
    WHITE = "\033[1;37;40m"
    GRAY = "\033[1;90;40m"


def show(args):
    logger = logging.getLogger(__name__)
    for plugin_path in args.plugins:
        plugin_name = Path(plugin_path).stem
        spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    network = load_network(
        {
            "model": args.model,
            "input_shape": args.input_shape,
            "input_format": args.input_format,
        }
    )

    print(Color.NONE + "======================================")
    print(Color.NONE + "LEGEND:")
    print(Color.NONE + " - " + Color.BLUE + "Droppable")
    print(Color.NONE + " - " + Color.YELLOW + "Linearizable")
    print(Color.NONE + " - " + Color.RED + "Rescalable")
    print(Color.NONE + " - " + Color.GREEN + "Droppable, Linearizable")
    print(Color.NONE + " - " + Color.PURPLE + "Droppable, Rescalable")
    print(Color.NONE + " - " + Color.CYAN + "Linearizable, Rescalable")
    print(Color.NONE + " - " + Color.WHITE + "Droppable, Linearizable, Rescalable")
    print(Color.NONE + " - " + Color.GRAY + "Not Modifiable")
    print(Color.NONE + "======================================\n")

    for i, layer in enumerate(network.layers):
        color = Color.GRAY
        if (
            isinstance(layer, Droppable)
            and isinstance(layer, Linearizable)
            and isinstance(layer, Rescalable)
        ):
            color = Color.WHITE
        elif isinstance(layer, Droppable) and isinstance(layer, Linearizable):
            color = Color.GREEN
        elif isinstance(layer, Droppable) and isinstance(layer, Rescalable):
            color = Color.PURPLE
        elif isinstance(layer, Linearizable) and isinstance(layer, Rescalable):
            color = Color.CYAN
        elif isinstance(layer, Droppable):
            color = Color.BLUE
        elif isinstance(layer, Linearizable):
            color = Color.YELLOW
        elif isinstance(layer, Rescalable):
            color = Color.RED
        print("\033[0mLayer %4d: %s%s" % (i, color, layer))
        print("\033[0m            %s -> %s" % (layer.input_shape, layer.output_shape))
    print(Color.NONE)
    print(Color.NONE + "======================================\n")

    print()
    pytorch_model = network.as_pytorch()
    print("Number of neurons:", pytorch_model.num_neurons())
    print("Number of parameters:", pytorch_model.num_parameters)
