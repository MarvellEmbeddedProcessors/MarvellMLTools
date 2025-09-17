# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Marvell.

"""
Generate input and calibration npz files.
"""
import argparse
import os
import sys
import json
import onnx
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model_onnx", type=str, required=True)
parser.add_argument("--input_json_file", type=str, required=False)
parser.add_argument("--input_npz_file", type=str,
                    required=False, default="input.npz")
parser.add_argument("--calib_npz_file", type=str,
                    required=False, default="calibration.npz")


def save_dict_to_npz(dict_to_save, npz_file_name):
    """Get Input dictionary to npz file"""
    np.savez(npz_file_name, **dict_to_save)


def get_input_dict(model_onnx, input_json_file):
    """Get input and shape dictionry from json file"""
    dtype = "float32"
    shape_d = {}
    inputs_d = {}
    calib_d = {}
    input_start = 0

    model = onnx.load(model_onnx)
    input_all = [node.name for node in model.graph.input]
    input_initializer = [node.name for node in model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))

    # Determine input shapes
    for _input in model.graph.input:
        input_name = _input.name
        if input_name in net_feed_input:
            input_shape = [
                d.dim_value for d in _input.type.tensor_type.shape.dim]
            if input_shape[0] == 0:
                print("Warning : Found a dynamic dimension in shape, resetting it to 1")
                input_shape[0] = 1
            if 0 in input_shape:
                print("Error: Invalid dimension in shape")
                sys.exit(1)
            shape_d[input_name] = input_shape

    if os.path.isfile(input_json_file):
        with open(input_json_file, "r", encoding='UTF-8') as json_in:
            input_json_obj = json.load(json_in)
        raw_input = np.asarray(input_json_obj["inputs"]).reshape(1, -1)

    for input_name in net_feed_input:
        input_shape = shape_d[input_name]
        input_size = np.prod(input_shape)
        input_end = input_start + input_size
        data = raw_input[0, input_start:input_end]
        inputs_d[input_name] = data.reshape(input_shape).astype(dtype)
        input_start = input_end

    calib_d = {"calib_data": [inputs_d]}

    return inputs_d, calib_d


if __name__ == "__main__":
    args = parser.parse_args()

    # json to npz Conversion
    inputs_dict, calib_dict = get_input_dict(
        args.model_onnx, args.input_json_file)
    save_dict_to_npz(inputs_dict, args.input_npz_file)
    save_dict_to_npz(calib_dict, args.calib_npz_file)
