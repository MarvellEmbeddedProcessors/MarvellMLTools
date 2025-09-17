# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Marvell.

''' File type conversion routines '''

import sys
import os
import json
import argparse
import numpy as np
import onnx


def _type_to_nptype(_elem_type: str):
    dtype_mapping = {
        "int8": np.int8,
        "uint8": np.uint8,
        "int16": np.int16,
        "uint16": np.uint16,
        "int32": np.int32,
        "uint32": np.uint32,
        "int64": np.int64,
        "uint64": np.uint64,
        "float16": np.float16,
        "float": np.float32
    }

    return dtype_mapping[_elem_type]


def flatten_list(nested_list):
    ''' Flatten nested list '''
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def json2npz(model_onnx: str, json_filename: str, npz_filename: str, io_type: str):
    ''' Convert JSON file to NPZ '''

    model = onnx.load(model_onnx)
    onnx_types = {key: value for value, key in onnx.TensorProto.DataType.items()}

    with open(json_filename, encoding='UTF-8') as json_file:
        json_data = json.load(json_file)

    initializers = [
        initializer.name for initializer in model.graph.initializer]

    # Flatten nested lists
    for k in json_data.keys():
        if isinstance(json_data[k][0], list):
            json_data[k] = flatten_list(json_data[k])

    # Fix keys in json_data
    keys = list(json_data.keys())
    _idx = 0
    if io_type == "input":
        for _input in model.graph.input:
            if _input.name in initializers:
                continue
            if _idx < len(keys):
                json_data[_input.name] = json_data.pop(keys[_idx])
                _idx += 1
    elif io_type == "output":
        for _output in model.graph.output:
            if _idx < len(keys):
                json_data[_output.name] = json_data.pop(keys[_idx])
                _idx += 1

    io_info = {}
    for _input in model.graph.input:
        if _input.name in initializers:
            continue
        io_info[_input.name] = onnx_types[_input.type.tensor_type.elem_type].lower()

    for _output in model.graph.output:
        io_info[_output.name] = onnx_types[_output.type.tensor_type.elem_type].lower()

    data = {}
    for key in json_data:
        if key not in io_info:
            print(f'Data {key} not present in JSON, Invalid type {io_type}')
            sys.exit()

        _dtype = _type_to_nptype(io_info[key])
        flat_data = json_data[key]
        graph_io = model.graph.input if io_type == "input" else model.graph.output
        shape_dims = next(
            (d.type.tensor_type.shape.dim for d in graph_io if d.name == key), None
        )
        if shape_dims is None:
            raise ValueError(f"'{key}' not found in the model graph.")
        shape_1 = [dim.dim_value for dim in shape_dims]
        total_elements = np.prod(shape_1)
        if len(flat_data) < total_elements:
            flat_data.extend([0] * (total_elements - len(flat_data)))
        elif len(flat_data) > total_elements:
            flat_data = flat_data[:total_elements]
        data[key] = np.array(flat_data, dtype=_dtype).reshape(shape_1)

    np.savez(npz_filename, **data)


def json2bin(model_onnx: str, json_filename: str, bin_filename: str, io_type: str):
    ''' Convert JSON file to bin '''

    model = onnx.load(model_onnx)
    onnx_types = {key: value for value, key in onnx.TensorProto.DataType.items()}

    with open(json_filename, encoding='UTF-8') as json_file:
        json_data = json.load(json_file)

    initializers = []
    for initializer in model.graph.initializer:
        initializers.append(initializer.name)

    # Fix keys in json_data
    keys = list(json_data.keys())
    _idx = 0
    if io_type == "input":
        for _input in model.graph.input:
            if _input.name in initializers:
                continue
            json_data[_input.name] = json_data.pop(keys[_idx])
            _idx = _idx + 1
    elif io_type == "output":
        for _output in model.graph.output:
            json_data[_output.name] = json_data.pop(keys[_idx])
            _idx = _idx + 1

    io_info = {}
    for _input in model.graph.input:
        if _input.name in initializers:
            continue
        io_info[_input.name] = onnx_types[_input.type.tensor_type.elem_type].lower()

    for _output in model.graph.output:
        io_info[_output.name] = onnx_types[_output.type.tensor_type.elem_type].lower()

    with open(bin_filename, "wb") as bin_file:
        for key in json_data:
            if key not in io_info:
                print(
                    f'Data {key} not present in JSON, Invalid type {io_type}')
                sys.exit()

            _dtype = _type_to_nptype(io_info[key])
            data_x = np.array(json_data[key], dtype=_dtype)
            bin_file.write(data_x.tobytes())

        bin_file.close()


def npz2json(model_onnx: str, npz_filename: str, json_filename: str, io_type: str):
    ''' Convert NPZ file to json '''

    model = onnx.load(model_onnx)
    npz_data = np.load(npz_filename)
    keys = npz_data.files

    json_data_raw = {}
    for key in keys:
        json_list = []
        for idx in range(npz_data[key].shape[0]):
            array = npz_data[key][idx, ...].tolist()
            json_list.append(array)

        json_data_raw[key] = json_list

    # Fix keys in json_data
    keys = list(json_data_raw.keys())
    _idx = 0
    json_data = {}
    if io_type == "input":
        for _input in model.graph.input:
            json_data[_input.name] = json_data_raw[keys[_idx]]
            _idx = _idx + 1
    elif io_type == "output":
        for _output in model.graph.output:
            json_data[_output.name] = json_data_raw[keys[_idx]]
            _idx = _idx + 1

    json_string = json.dumps(json_data, indent=4)
    with open(json_filename, "w", encoding='UTF-8') as json_file:
        json_file.write(json_string)
        json_file.close()


def bin2json(model_onnx: str, bin_filename: str, json_filename: str, io_type: str):
    ''' Convert bin file to json '''

    model = onnx.load(model_onnx)
    onnx_types = {key: value for value, key in onnx.TensorProto.DataType.items()}

    io_info = {}
    if io_type == "input":
        for _input in model.graph.input:
            io_info[_input.name] = onnx_types[_input.type.tensor_type.elem_type].lower()
    elif io_type == "output":
        for _output in model.graph.output:
            io_info[_output.name] = onnx_types[_output.type.tensor_type.elem_type].lower()

    json_data = {}
    _offset = 0
    for key, _elem_type in io_info.items():
        graph_io = model.graph.input if io_type == "input" else model.graph.output
        io_shape = next(
            (d.type.tensor_type.shape.dim for d in graph_io if d.name == key), None
        )
        if io_shape is None:
            raise ValueError(f"'{key}' not found in the model graph.")
        io_shape = [dim.dim_value for dim in io_shape]
        _dtype = _type_to_nptype(_elem_type)
        data_raw = np.fromfile(bin_filename, dtype=_dtype,
                               count=np.prod(io_shape), offset=_offset)
        _offset = _offset + np.prod(io_shape) * np.dtype(_dtype).itemsize
        json_data[key] = data_raw.reshape(io_shape).tolist()

    json_string = json.dumps(json_data, indent=4)
    with open(json_filename, "w", encoding='UTF-8') as json_file:
        json_file.write(json_string)
        json_file.close()


def npz2bin(model_onnx: str, npz_filename: str, bin_filename: str, io_type: str):
    ''' Convert NPZ file to bin '''

    tmp_json_file = 'tmp.json'
    npz2json(model_onnx=model_onnx,
             npz_filename=npz_filename, json_filename=tmp_json_file, io_type=io_type)
    json2bin(model_onnx=model_onnx,
             json_filename=tmp_json_file, bin_filename=bin_filename, io_type=io_type)
    os.remove(tmp_json_file)


def bin2npz(model_onnx: str, bin_filename: str, npz_filename: str, io_type: str):
    ''' Convert bin file to npz '''
    tmp_json_file = 'tmp.json'
    bin2json(model_onnx=model_onnx, bin_filename=bin_filename,
             json_filename=tmp_json_file, io_type=io_type)
    json2npz(model_onnx=model_onnx, json_filename=tmp_json_file,
             npz_filename=npz_filename, io_type=io_type)
    os.remove(tmp_json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Conversion funnctions")
    parser.add_argument("--model_onnx", type=str, action='store',
                        help="ONNX model file")
    parser.add_argument("--json_filename", type=str,
                        action='store', help="JSON file name")
    parser.add_argument("--bin_filename", type=str,
                        action='store', help="Binary file name")
    parser.add_argument("--npz_filename", type=str,
                        action='store', help="NPZ file name")
    parser.add_argument("--io_type", type=str, action='store',
                        help="File type input/output")

    if sys.argv[1] == "json2npz":
        args = parser.parse_args(sys.argv[2:])
        json2npz(model_onnx=args.model_onnx, json_filename=args.json_filename,
                 npz_filename=args.npz_filename, io_type=args.io_type)
    elif sys.argv[1] == "json2bin":
        args = parser.parse_args(sys.argv[2:])
        json2bin(model_onnx=args.model_onnx, json_filename=args.json_filename,
                 bin_filename=args.bin_filename, io_type=args.io_type)
    elif sys.argv[1] == "npz2json":
        args = parser.parse_args(sys.argv[2:])
        npz2json(model_onnx=args.model_onnx, npz_filename=args.npz_filename,
                 json_filename=args.json_filename, io_type=args.io_type)
    elif sys.argv[1] == "bin2json":
        args = parser.parse_args(sys.argv[2:])
        bin2json(model_onnx=args.model_onnx, bin_filename=args.bin_filename,
                 json_filename=args.json_filename, io_type=args.io_type)
    elif sys.argv[1] == "npz2bin":
        args = parser.parse_args(sys.argv[2:])
        npz2bin(model_onnx=args.model_onnx, npz_filename=args.npz_filename,
                bin_filename=args.bin_filename, io_type=args.io_type)
    elif sys.argv[1] == "bin2npz":
        args = parser.parse_args(sys.argv[2:])
        bin2npz(model_onnx=args.model_onnx, bin_filename=args.bin_filename,
                npz_filename=args.npz_filename, io_type=args.io_type)
    else:
        raise NameError("Invalid conversion function")
