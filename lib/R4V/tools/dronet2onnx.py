#!/usr/bin/env python
import argparse
import numpy as np
import onnx
import onnxmltools
import os

from keras.models import model_from_json
from onnx import helper


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the dronet network to an onnx model."
    )

    parser.add_argument(
        "model_path", type=str, help="path to the keras model (in json format)"
    )
    parser.add_argument(
        "weights_path",
        type=str,
        help="path to the keras model weights (in hdf5 format)",
    )
    parser.add_argument(
        "output_path", type=str, help="where to write the resulting onnx model"
    )
    return parser.parse_args()


def concat_outputs(model):
    node_map = {}
    var_map = {}
    for node in model.graph.node:
        assert len(node.output) == 1
        for output_name in node.output:
            node.name = output_name
            if node.op_type not in ["Constant"]:
                node_map[output_name] = node
            else:
                var_map[output_name] = node
    for initializer in model.graph.initializer:
        var_map[initializer.name] = initializer

    nodes = []
    visited = set()

    def topo_sort(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        for name in node.input:
            if name in node_map:
                topo_sort(node_map[name])
        nodes.append(node)

    for node in node_map.values():
        topo_sort(node)

    for node in nodes:
        print(node.op_type, node.input, "->", node.output)
    print("===============================================\n")

    new_nodes = nodes

    new_nodes.append(
        helper.make_node(
            "Concat",
            inputs=[n.name for n in model.graph.output],
            outputs=["output"],
            axis=1,
        )
    )

    print("\n===================")
    for node in new_nodes:
        print(node.op_type, node.input, node.output)

    new_output = [
        helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, ("N", 2))
    ]
    graph = helper.make_graph(
        new_nodes, "dronet", model.graph.input, new_output, model.graph.initializer
    )
    model = helper.make_model(graph)

    return model


def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.model_path) as json_file:
        keras_model = model_from_json(json_file.read())
    keras_model.load_weights(args.weights_path)

    onnx_model = onnxmltools.convert_keras(keras_model)
    optimized_onnx_model = concat_outputs(onnx_model)
    onnxmltools.utils.save_model(optimized_onnx_model, args.output_path)


if __name__ == "__main__":
    main(_parse_args())
