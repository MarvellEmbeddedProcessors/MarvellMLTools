# SPDX-License-Identifier: Apache-2.0

#Copyright (c) 2024 Marvell.

# Based on lucid_dataset_parser.py from github.com/doriguzzi/lucid-ddos repo
# https://github.com/doriguzzi/lucid-ddos/blob/master/lucid_dataset_parser.py

import os
import shutil
import csv
import pprint
import numpy as np
import pyshark
import argparse
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from util_functions import *
from lucid_dataset_parser import process_live_traffic, parse_labels, dataset_to_list_of_fragments, normalize_and_padding
import time
import ctypes


class colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


PREDICT_HEADER = ['Model', 'Time', 'Packets', 'Samples', 'DDOS%',
                  'Accuracy', 'F1Score', 'TPR', 'FPR', 'TNR', 'FNR', 'Source']

# Function to print colored text


def print_colored(text, color):
    print(color + text + colors.END)

# Function to print confusion matrix


def report_results(Y_true, Y_pred, packets, model_name, data_source, prediction_time, writer, iteration):
    ddos_rate = '{:04.3f}'.format(sum(Y_pred) / Y_pred.shape[0])

    if Y_true is not None and len(Y_true.shape) > 0:
        accuracy = accuracy_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred)
        tp, fn, fp, tn = confusion_matrix(
            Y_true, Y_pred, labels=[1, 0]).ravel()

        cm = confusion_matrix(Y_true, Y_pred, labels=[1, 0])
        print()
        print(f"Confusion Matrix:\n{cm}")
        print()

        tnr = tn / (tn + fp) if (tn + fp) > 0 else float(0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float(0)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float(0)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float(0)

        row = {'Model': model_name, 'Time': '{:04.3f}'.format(prediction_time), 'Packets': packets,
               'Samples': Y_pred.shape[0], 'DDOS%': ddos_rate, 'Accuracy': '{:05.4f}'.format(accuracy), 'F1Score': '{:05.4f}'.format(f1),
               'TPR': '{:05.4f}'.format(tpr), 'FPR': '{:05.4f}'.format(fpr), 'TNR': '{:05.4f}'.format(tnr), 'FNR': '{:05.4f}'.format(fnr), 'Source': data_source}
    else:
        row = {'Model': model_name, 'Time': '{:04.3f}'.format(prediction_time), 'Packets': packets,
               'Samples': Y_pred.shape[0], 'DDOS%': ddos_rate, 'Accuracy': "N/A", 'F1Score': "N/A",
               'TPR': "N/A", 'FPR': "N/A", 'TNR': "N/A", 'FNR': "N/A", 'Source': data_source}

    pprint.pprint(row, sort_dicts=False)
    writer.writerow(row)


def main():

    parser = argparse.ArgumentParser(
        description='DDoS attacks detection with convolutional neural networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-pl', '--pcap_file', nargs='?', type=str,
                        help='Perform a prediction on pcap file')

    parser.add_argument('-m', '--model', type=str,
                        help='File containing the model')

    parser.add_argument('-y', '--dataset_type', default=None, type=str,
                        help='Type of the dataset. Available options are: DOS2017, DOS2018, DOS2019, SYN2020')

    args = parser.parse_args()

    try:
        shutil.rmtree("predicted_data")
    except OSError as e:
        print(f"Error: predicted_data : {e.strerror}")

    os.makedirs("predicted_data")
    OUTPUT_FOLDER = "predicted_data/"

    pyargs = ["mlperf"]
    pyargs_bytes = []
    num_threads = 1
    for i in range(len(pyargs)):
        pyargs_bytes.append(bytes(pyargs[i], 'utf-8'))
    pyargs_array = (ctypes.c_char_p * (len(pyargs_bytes)+1))()
    pyargs_array[:-1] = pyargs_bytes

    global mldpc
    mldpc = ctypes.CDLL("/usr/lib/libmldpc.so")
    int_res = mldpc.mrvl_ml_init(1, pyargs_array, num_threads + 1)
    if int_res != 0:
        print("Initialization failed")
        exit(-1)

    if args.pcap_file is not None:
        predict_file = open(OUTPUT_FOLDER + 'predictions-' +
                            time.strftime("%Y%m%d-%H%M%S") + '.csv', 'a', newline='')
        # clean the file content (as we open the file in append mode)
        predict_file.truncate(0)
        predict_writer = csv.DictWriter(
            predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader()
        predict_file.flush()
        print("Running live data preprocessing")
        if args.pcap_file.endswith('.pcap'):
            pcap_file = args.pcap_file
            cap = pyshark.FileCapture(pcap_file)
            data_source = pcap_file.split('/')[-1].strip()
        else:
            print("Please specify a valid pcap file!")
            exit(-1)

        print("Preprocessing on network traffic from: ", data_source)

        # load the labels, if available
        labels = parse_labels(args.dataset_type)

        if args.model is not None and args.model.endswith('.bin'):
            model_path = args.model
        else:
            print("No valid model specified!")
            exit(-1)

        model_filename = model_path.split('/')[-1].strip()
        filename_prefix = model_filename.split('n')[0] + 'n-'
        time_window = int(filename_prefix.split('t-')[0])
        max_flow_len = int(filename_prefix.split('t-')[1].split('n-')[0])
        model_name_string = model_filename.split(
            filename_prefix)[1].strip().split('.')[0].strip()

        # load ml model
        f1 = open(model_path, 'rb')
        f1.seek(0, os.SEEK_END)
        model_size = f1.tell()
        f1.close()
        f = open(model_path, 'rb')
        model_buffer = f.read(model_size)
        f.close()
        model_id = mldpc.mrvl_ml_model_load(model_buffer, model_size)
        if model_id < 0:
            print("Model load failed")
            exit(-1)

        input_dims = [1, 1, 10, 11]
        output_dims = [1]

        mldpc.mrvl_ml_io_alloc.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]

        # dequantized input buffer
        input_dsize = ctypes.c_ulonglong()
        input_dalloc = mldpc.mrvl_ml_io_alloc
        input_dalloc.restype = np.ctypeslib.ndpointer(
            dtype=np.float32, shape=input_dims)
        input_dbuffer = input_dalloc(model_id, 1, ctypes.byref(input_dsize))

        # quantized input buffer
        input_qsize = ctypes.c_ulonglong()
        input_qalloc = mldpc.mrvl_ml_io_alloc
        input_qalloc.restype = np.ctypeslib.ndpointer(
            dtype=np.int16, shape=input_dims)
        input_qbuffer = input_dalloc(model_id, 0, ctypes.byref(input_qsize))

        # quantized output buffer
        output_qsize = ctypes.c_ulonglong()
        output_qalloc = mldpc.mrvl_ml_io_alloc
        output_qalloc.restype = np.ctypeslib.ndpointer(
            dtype=np.int16, shape=output_dims)
        output_qbuffer = input_dalloc(model_id, 2, ctypes.byref(output_qsize))

        # dequantized output buffer
        output_dsize = ctypes.c_ulonglong()
        output_dalloc = mldpc.mrvl_ml_io_alloc
        output_dalloc.restype = np.ctypeslib.ndpointer(
            dtype=np.float32, shape=output_dims)
        output_dbuffer = input_dalloc(model_id, 3, ctypes.byref(output_dsize))

        mins, maxs = static_min_max(time_window)
        window_counter = 0
        mismatch_outer = 0
        malicious_outer = 0
        benign_outer = 0
        total_inference_complete = 0

        while (True):
            samples = process_live_traffic(
                cap, args.dataset_type, labels, max_flow_len, traffic_type="all", time_window=time_window)
            if len(samples) > 0:
                X, Y_true, keys = dataset_to_list_of_fragments(samples)
                X = np.array(normalize_and_padding(
                    X, mins, maxs, max_flow_len))
                if labels is not None:
                    Y_true = np.array(Y_true)
                else:
                    Y_true = None

                X_val = np.expand_dims(X, axis=3)
                total_infer = X_val.shape[0]
                pt0 = time.time()
                Y_pred = np.zeros(total_infer)
                mismatch = 0
                malicious = 0
                benign = 0

                for count in range(0, total_infer):
                    input_dbuffer = X_val[count].astype(
                        'float32').reshape(input_dims)
                    mldpc.mrvl_ml_model_quantize(model_id, ctypes.c_void_p(
                        input_dbuffer.ctypes.data), ctypes.c_void_p(input_qbuffer.ctypes.data))
                    mldpc.mrvl_ml_model_run(model_id, ctypes.c_void_p(
                        input_qbuffer.ctypes.data), ctypes.c_void_p(output_qbuffer.ctypes.data), 1)
                    mldpc.mrvl_ml_model_dequantize(model_id, ctypes.c_void_p(
                        output_qbuffer.ctypes.data), ctypes.c_void_p(output_dbuffer.ctypes.data))
                    output_arr = output_dbuffer.tolist()

                    if output_arr[0] > 0.5:
                        label = 1
                        malicious += 1
                        malicious_outer += 1
                        Y_pred[count] = 1
                    else:
                        label = 0
                        benign += 1
                        benign_outer += 1
                        Y_pred[count] = 0
                    if Y_true is not None:
                        if label != Y_true[count]:
                            mismatch += 1
                            mismatch_outer += 1

                ddos_rate = '{:04.1f} %'.format(
                    (sum(Y_pred) / Y_pred.shape[0])*100)
                if Y_true is not None:
                    correct = total_infer - mismatch
                    print(
                        f"{correct} out of {total_infer} inferences were identified correctly")
                    print(
                        f"{malicious} out of {total_infer} inferences were predicted as malicious")
                    accuracy = (correct / total_infer) * 100
                    print_colored("***************************", colors.GREEN)
                    print_colored(
                        f"NUM INFERENCES: {total_infer}", colors.GREEN)
                    print_colored(f"ACCURACY: {accuracy}", colors.GREEN)
                    print_colored("***************************", colors.GREEN)
                else:
                    if malicious >= benign:
                        print(
                            f"{malicious} of {total_infer} inferences were predicted as malicious")
                        print_colored(
                            "****************************************************", colors.RED)
                        print_colored(
                            f"Probablity of DDOS attack {ddos_rate}", colors.RED)
                        print_colored(
                            "****************************************************", colors.RED)
                        print_colored(
                            "WARNING: The system is possibly under a DDOS attack", colors.RED)
                        print_colored(
                            "****************************************************", colors.RED)
                    else:
                        print(
                            f"{malicious} out of {total_infer} inferences were predicted as malicious")
                        print_colored(
                            "***********************************", colors.GREEN)
                        print_colored(
                            f"Probablity of DDOS attack {ddos_rate}", colors.GREEN)
                        print_colored(
                            "The system health is mostly normal", colors.GREEN)
                        print_colored(
                            "***********************************", colors.GREEN)

                pt1 = time.time()
                prediction_time = pt1 - pt0

                [packets] = count_packets_in_dataset([X])
                print(f"Packets in dataset: {packets}")
                window_counter += 1
                report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string,
                               data_source, prediction_time, predict_writer, window_counter)
                print()
                predict_file.flush()
                total_inference_complete += total_infer

            elif isinstance(cap, pyshark.FileCapture) == True:
                print("\nNo more packets in file ", data_source)
                break

        predict_file.close()


if __name__ == "__main__":
    main()
