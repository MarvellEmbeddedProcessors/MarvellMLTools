# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Marvell.

"""
Convert a model with dynamic shapes to static shapes.
"""

import argparse
import onnx

parser = argparse.ArgumentParser()
parser.add_argument("--input_onnx", type=str, required=True)
parser.add_argument("--output_onnx", type=str, required=False)


def dynamic_to_static_shape(input_onnx, output_onnx):
    """Convert dynamic shape to static shape"""

    model = onnx.load(input_onnx)

    nb_inputs = len(model.graph.input)
    for i in range(0, nb_inputs):
        nb_dims = len(model.graph.input[i].type.tensor_type.shape.dim)
        for j in range(0, nb_dims):
            if model.graph.input[i].type.tensor_type.shape.dim[j].dim_value == 0:
                model.graph.input[i].type.tensor_type.shape.dim[j].dim_value = 1

    nb_outputs = len(model.graph.output)
    for i in range(0, nb_outputs):
        nb_dims = len(model.graph.output[i].type.tensor_type.shape.dim)
        for j in range(0, nb_dims):
            if model.graph.output[i].type.tensor_type.shape.dim[j].dim_value == 0:
                model.graph.output[i].type.tensor_type.shape.dim[j].dim_value = 1

    onnx.save(model, output_onnx)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.output_onnx is None:
        args.output_onnx = args.input_onnx

    dynamic_to_static_shape(args.input_onnx, args.output_onnx)
