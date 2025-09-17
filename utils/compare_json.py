# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Marvell.

"""Output Json Comparison Module"""

import sys
import json
import argparse
import numpy as np

# Order of Magnitude
OOM_LIMIT = 100
FP16_RES_TABLE = {i: np.power(2.0, i) for i in range(-24, 17)}

def fp16_local_resolution(x):
    """Calculate the resolution of a fp16 value"""
    x = np.abs(x)
    for i in range(-13, 17):
        if x < FP16_RES_TABLE[i]:
            return FP16_RES_TABLE[i - 11]
    print(f"error in determening the resolution of fp16 value : {x}")
    sys.exit(0)

def calculate_relative_difference(tensor_1, tensor_2):
    """Calculate the relative difference between 2 tensors"""
    diff = np.zeros(tensor_1.shape, dtype=float)
    for i in range(tensor_1.shape[0]):
        if tensor_2[i] == 0:
            if tensor_1[i] == 0:
                diff[i] = 0
            else:
                diff[i] = np.nan
        else:
            diff[i] = (tensor_1[i] - tensor_2[i]) / tensor_2[i] * 100
    return diff

def calculate_bit_resolution(max_output, max_reference, quantize="int8"):
    """Calculate the bit resolution of the output tensor"""
    count_unit = None
    if quantize == "uint8":
        if max_output == 0 and max_reference != 0:
            count_unit = max_reference / 225
        else:
            count_unit = max_output / 255
    elif quantize == "int8" or quantize == "pq":
        if max_output == 0 and max_reference != 0:
            count_unit = max_reference / 127
        else:
            count_unit = max_output / 127
    return count_unit

def print_tensor_info(print_level, tensor_1, tensor_2, diff, match_oom_limits, match, additional_info=None):
    """Helper function to print tensor comparison information."""
    header = (
        f"{'Num':^5},  {'Test Tensor':>15},  {'Base Tensor':>15},  "
        f"{'Difference':>15},  {'% Off':>12},  "
    )

    if additional_info:
        header += f"{additional_info['header']},  "

    header += f"{'match_oom_limits':>15},  {'Match':>7}"
    if print_level == "full" or (print_level == "diff" and match.sum() < match_oom_limits.sum()):
        print(header, flush=True)

    for i, ele in enumerate(tensor_2):
        if print_level == "full" or (print_level == "diff" and match_oom_limits[i] and not match[i]):
            row = (
                f"{i:5},  {tensor_1[i]:+15.8f},  {ele:+15.8f},  "
                f"{tensor_1[i]-tensor_2[i]:+15.8f},  {diff[i]:+10.5f}%,  "
            )

            if additional_info:
                row += f"{additional_info['values'][i]:{additional_info['format']}},  "

            row += f"{match_oom_limits[i]:>15},  {match[i]:>7}"
            print(row, flush=True)

def compare_1d_tensors(tensor_1, tensor_2, quantize, fgft, delta, print_level):
    """Compare 2 1d tensor arrays"""

    data_match = False
    if tensor_1.shape == tensor_2.shape:

        # Initialize variables
        bit_off = np.zeros(tensor_1.shape, dtype=np.int32)
        oom_reference = np.zeros(tensor_2.shape[0], dtype=float)
        match = np.zeros(tensor_1.shape, dtype=bool)
        match_rel_diff = np.zeros(tensor_1.shape[0], dtype=bool)
        match_bit_off = np.zeros(tensor_1.shape[0], dtype=bool)
        match_oom_limits = np.zeros(tensor_1.shape[0], dtype=bool)
        max_output = np.abs(tensor_1).max()
        max_reference = np.abs(tensor_2).max()
        oom_reference.fill(OOM_LIMIT)

        # Max absolute differnce
        max_abs_diff = max(abs(ele1 - ele2) for ele1, ele2 in zip(tensor_1, tensor_2))
        diff = calculate_relative_difference(tensor_1, tensor_2)

        # Calculate the bit resolution of the output tensor
        count_unit = calculate_bit_resolution(max_output, max_reference, quantize)

        # oom calculation
        for i in range(tensor_1.shape[0]):
            if tensor_2[i] != 0.0:
                oom_reference[i] = np.abs(max_reference / tensor_2[i])
            elif tensor_1[i] != 0.0:
                oom_reference[i] = np.abs(max_reference / tensor_1[i])

        # Datamatching tests
        exp_min = tensor_2 * (1.0 - np.sign(tensor_2) * fgft)
        exp_max = tensor_2 * (1.0 + np.sign(tensor_2) * fgft)

        match_rel_diff = (tensor_1 <= exp_max) & (tensor_1 >= exp_min)
        match_oom_limits = oom_reference <= OOM_LIMIT

        if quantize == "fp16":
            match = match_oom_limits & match_rel_diff
        else:
            if quantize == "pq":
                bit_off = np.abs(np.round((tensor_1-tensor_2)/count_unit))
            else:
                bit_off = (np.ceil(np.abs(tensor_1 - tensor_2) / (count_unit / 2.0)) - 1)
                # Set exact match cases to 0
                bit_off = (tensor_1 != tensor_2) * bit_off
            match_bit_off = bit_off <= delta
            match = (match_oom_limits & match_rel_diff) | match_bit_off
            n_match_bit_off = match_bit_off.sum()

        n_match_rel_diff = match_rel_diff.sum()
        n_match_oom_limits = match_oom_limits.sum()
        n_match = match.sum()

        min_rel_diff = np.nanmin(diff[match_oom_limits])
        max_rel_diff = np.nanmax(diff[match_oom_limits])

        # Data Matching Report
        np.set_printoptions(suppress=True)

        print("\nDEBUG: info about the output vector")
        print("Test result")
        print(f"Average {tensor_1.mean()}, min {tensor_1.min()}, max {tensor_1.max()}")
        print("Base result")
        print(f"Average {tensor_2.mean()}, min {tensor_2.min()}, max {tensor_2.max()}")

        print("\nData_Matching_Report")
        print(f"Total entries {tensor_1.shape[0]}")
        print(f"MSE: {((tensor_2 - tensor_1) ** 2).mean()}")
        print(f"max_abs_diff: {max_abs_diff}")
        print(f"max_rel_diff: {max(abs(diff[match_oom_limits])):.6f}%")
        print(f"Fudge factor criteria is {fgft * 100}%")
        print(
            f"Max - Fudge off is {min_rel_diff:+10.5f}%, "
            f"Num: {np.where(diff == min_rel_diff)[0]}"
        )
        print(
            f"Max + Fudge off is {max_rel_diff:+10.5f}%, "
            f"Num: {np.where(diff == max_rel_diff)[0]}"
        )
        if quantize != "fp16":
            print(f"Bit off criteria is {delta}")
            print(f"1 bit unit in float is {count_unit}".format())
            print(f"Max bit off is {bit_off.max()} , Num: {np.where(bit_off == bit_off.max())[0]}")
            print(f"match based on result within range {n_match_rel_diff}")
            print(f"match based on bit off {n_match_bit_off}")
        print(f"OOM limit Criteria {OOM_LIMIT}")
        print(f"match based on oom limit {n_match_oom_limits}")
        print(f"match overall {n_match}")

        # Print Table
        if quantize == "fp16":
            if tensor_2.max() >= 65504.0 or tensor_2.min() <= -65504.0:
                print("\nWARNING: SATURATION ERROR, PLEASE CHECK THE MODEL")

            additional_info = None
            print_tensor_info(
                print_level, tensor_1, tensor_2, diff, match_oom_limits, match, additional_info
            )
        else:
            additional_info = {
                'header': 'Bit off',
                'values': bit_off,
                'format': '+8.5f'
            }
            print_tensor_info(
                print_level, tensor_1, tensor_2, diff, match_oom_limits, match, additional_info
            )

        # Pass / Fail
        for i in range(tensor_1.shape[0]):
            if match_oom_limits[i]:
                if match[i]:
                    data_match = True
                else:
                    data_match = False
                    break

        if data_match:
            print("\nData Matching: PASSED\n", flush=True)
        else:
            print("\nData Matching: FAILED\n", flush=True)
    else:
        print("tensor_1.shape != tensor_2.shape")
        print("\nData Matching: FAILED\n", flush=True)

    return data_match

def compare_json_files(test_json_file, base_json_file, quantize, fgft, delta, print_level):
    """Compare 2 json lists"""
    print(f"Comparing {test_json_file} vs {base_json_file}")

    def get_json_data(json_file):
        """Get output tensor from a json file"""
        print(f"json.load({json_file})...", flush=True)
        with open(json_file, "r") as json_in_f:
            json_data = json.load(json_in_f)
        return json_data

    test_json_data = get_json_data(test_json_file)
    base_json_data = get_json_data(base_json_file)

    for key in test_json_data:
        test_array = np.asarray(test_json_data[key]).flatten()
        base_array = np.asarray(base_json_data[key]).flatten()
        compare_1d_tensors(test_array, base_array, quantize, fgft, delta, print_level)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json_file", type=str, required=True)
    parser.add_argument("--base_json_file", type=str, required=True)
    parser.add_argument(
        "--quantize",
        type=str,
        required=False,
        default="fp16",
        choices=["fp16", "int8", "pq"],
        help="quantization type used for compilation"
    )
    parser.add_argument(
        "-fgft", "--fudge_factor", type=float, required=False, default=0.03, help="fudge_factor"
    )
    parser.add_argument(
        "-d", "--bit_off_delta", type=float, required=False, default=3.0, help="bit_off_criteria"
    )
    parser.add_argument(
        "--print_level",
        required=False,
        default=None,
        type=str,
        choices=[None, "full", "diff"],
        help="Optional: Printing a pretty table for output comparison",
    )
    args = parser.parse_args()


    compare_json_files(
        args.test_json_file,
        args.base_json_file,
        args.quantize,
        args.fudge_factor,
        args.bit_off_delta,
        args.print_level,
    )
