#!/usr/bin/env python

# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Marvell.

# Convert output.bin to output.json #

import sys
import json
import numpy as np
import argparse


def get_cmd_line_options():
    parser = argparse.ArgumentParser(
        description="Convert output raw fp32 bin file to json file")
    parser.add_argument("-b", "--output_bin", type=str,
                        required=True, help="output raw fp32 bin filename (e.g., output.bin).")
    parser.add_argument("-j", "--output_json", type=str,
                        required=False, default=None,
                        help="Optional: output-bin file in json filename (default: "
                             "use same name as output bin file but use json as file extension.")
    parser.add_argument("-t", "--type", type=str,
                        required=False, default="fp32",
                        help="Optional: element type in the raw output bin file "
                             "-- fp32|fp16|uint8|int8|uint16|int16 (default: fp32).")

    return parser


if __name__ == '__main__':
    parser = get_cmd_line_options()
    args = parser.parse_args()

    output_bin_filename = args.output_bin

    output_json_filename = args.output_json
    if args.output_json is None:
        output_json_filename = output_bin_filename[:-3] + 'json'

    output_bin_elem_type = "float32"
    if args.type != "fp32":
        if args.type in ["fp32", "fp16", "uint8", "int8", "uint16", "int16"]:
            output_bin_elem_type = args.type
        else:
            parser.print_help(sys.stderr)
            exit(0)

    print("Converting {} to {}...".format(
        output_bin_filename, output_json_filename))

    # read raw fp32 output bin file
    output_raw = np.fromfile(output_bin_filename, dtype=output_bin_elem_type)
    output_fp32 = list()
    for idx in range(np.size(output_raw)):
        output_fp32.append(output_raw[idx])
    output_json_str = "{}".format(output_fp32)
    output_json_str = "{" + ' "outputs": [ {} ] '.format(output_json_str) + "}"

    with open(output_json_filename, 'w+') as json_f:
        json_f.write(output_json_str)

    with open(output_json_filename, 'r') as json_f:
        tmp_json = json.load(json_f)
