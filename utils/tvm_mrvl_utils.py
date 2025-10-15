import os
import sys
import onnx
import requests
import random
import time
import glob
import json
import torch
from onnx.tools import update_model_dims
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, shape_inference

# Dynamic to Static Batch
def set_fixed_batch_size(
    model_path: str, output_path: str, batch_size: int
):
    """
    Sets a fixed batch size for an ONNX model.

    This function identifies inputs and outputs with a dynamic first dimension
    (typically the batch size) and sets it to the specified value.

    Args:
        model_path (str): Path to the input ONNX model.
        output_path (str): Path to save the modified ONNX model.
        batch_size (int): The fixed batch size to set.
    """
    try:
        model = onnx.load(model_path)

        # Create dictionaries to hold the new input and output shapes
        input_dims_new = {}
        output_dims_new = {}

        # print oiginal dims
        # for inp in model.graph.input:
            # print(f" {inp.name} Shape {inp.type.tensor_type.shape.dim}")
        # Update input dimensions
        for inp in model.graph.input:
            dims = inp.type.tensor_type.shape.dim
            if dims and (dims[0].dim_value <= 0 or dims[0].dim_param):
                rest_dims = [d.dim_param if d.dim_param else d.dim_value for d in dims[1:]]
                new_shape = [batch_size] + rest_dims
                input_dims_new[inp.name] = new_shape
                # print(f"  - Changing1 '{inp.name}' to {new_shape}")

        # Update output dimensions
        # print("\nUpdating output shapes...")
        for out in model.graph.output:
            dims = out.type.tensor_type.shape.dim
            if dims and (dims[0].dim_value <= 0 or dims[0].dim_param):
                rest_dims = [d.dim_param if d.dim_param else d.dim_value for d in dims[1:]]
                new_shape = [batch_size] + rest_dims
                output_dims_new[out.name] = new_shape
                # print(f"  - Changing '{out.name}' to {new_shape}")

        if not input_dims_new and not output_dims_new:
            # print("\nNo dynamic batch dimensions found to update.")
            onnx.save(model, output_path)
            return

        # Create the updated model
        updated_model = update_model_dims.update_inputs_outputs_dims(
            model, input_dims_new, output_dims_new
        )

        # Save the updated model
        onnx.save(updated_model, output_path)
        # print(f"\nModel with fixed batch size saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def get_resnet50_sample_image(image_path="./imagenet_images/n02110341_10112.jpeg", input_npz_path='input.npz'):
    """
    Downloads a sample image, preprocesses it for ResNet-50,
    and saves it as an .npz file for TVMC.
    """
    # 1 sample image URL (a picture of a golden retriever)
    img = Image.open(image_path).resize((224, 224))
    input_name = "data"
    
    # 2. Convert to NumPy array and scale to [0, 1]
    image_array = np.array(img).astype('float32') / 255.0
    
    # 3. Transpose from (H, W, C) to (C, H, W)
    transposed_array = np.transpose(image_array, (2, 0, 1))
    
    # 4. Normalize using ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    # Reshape mean and std to (C, 1, 1) to broadcast correctly
    normalized_array = (transposed_array - mean[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]
    
    # 5. Add a batch dimension to make it (N, C, H, W)
    input_tensor = np.expand_dims(normalized_array, axis=0)
    
    # 6. Save as a .npz file
    np.savez(input_npz_path, **{input_name: input_tensor})
    # print(f"✅ Successfully created '{input_npz_path}'")
    return img
    
def run_onnxrt_inference(model_path, input_npz_path, output_npz_path):
    """
    Runs inference using ONNX Runtime and saves the output to an .npz file.
    """
    

    # Load the model
    session = ort.InferenceSession(model_path)
    
    # Load input data
    input_data = np.load(input_npz_path)
    input_name = session.get_inputs()[0].name
    input_tensor = input_data[input_name].astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {input_name: input_tensor})
    
    # Save output to .npz file
    np.savez(output_npz_path, **{session.get_outputs()[0].name: outputs[0]})
    # print(f"✅ Successfully created '{output_npz_path}'")

class ResNet50DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = ort.InferenceSession(model_path, None)
        self.input_name = session.get_inputs()[0].name
        self.height = 224
        self.width = 224
    
        # Convert image to input data
        self.nhwc_data_list = self._preprocess_images(
            calibration_image_folder, size_limit=0
        )
        
        self.datasize = len(self.nhwc_data_list)

    def _preprocess_images(self, images_folder: str, size_limit=0):
        """
        Loads a batch of images and preprocess them
        parameter images_folder: path to folder storing images
        parameter size_limit: number of images to load. Default is 0 which means all images are picked.
        return: list of matrices characterizing multiple images
        """
        image_names = os.listdir(images_folder)
        if size_limit > 0 and len(image_names) >= size_limit:
            batch_filenames = [image_names[i] for i in range(size_limit)]
        else:
            batch_filenames = image_names
        unconcatenated_batch_data = []

        for image_name in batch_filenames:
            image_filepath = images_folder + "/" + image_name
            # pillow_img = Image.new("RGB", (self.width, self.height))
            # pillow_img.paste(Image.open(image_filepath).resize((self.width, self.height)))
            # input_data = np.float32(pillow_img) - np.array(
            #     [123.68, 116.78, 103.94], dtype=np.float32
            # )
            # nhwc_data = np.expand_dims(input_data, axis=0)
            # nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard
            # unconcatenated_batch_data.append(nchw_data)
            img = Image.open(image_filepath).resize((224, 224))
            input_name = "data"
            image_array = np.array(img).astype('float32') / 255.0
            transposed_array = np.transpose(image_array, (2, 0, 1))
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized_array = (transposed_array - mean[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]
            input_tensor = np.expand_dims(normalized_array, axis=0)
            unconcatenated_batch_data.append(input_tensor)
        batch_data = np.concatenate(
            np.expand_dims(unconcatenated_batch_data, axis=0), axis=0
        )
        return batch_data

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

def create_qdq_model(input_model_path, output_model_path):
    """
    Creates a QDQ (Quantize-Dequantize) model from the given ONNX model.
    If input_bin_path is provided, it will be used for calibration data.
    """
    shape_inference.quant_pre_process(input_model_path, output_model_path)
    calib_image_folder = "/home/shubhaml/tvm_devel_3/tvm/3rdparty/marvell/tvm-misc/demos/imagenet_dataset"
    calibration_data_reader = ResNet50DataReader(
        calib_image_folder, input_model_path
    )
    
    quantize_static(
            output_model_path,
            output_model_path,
            calibration_data_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=False,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "MatMul", "Add", "Mul", "Relu", "Gemm",
                                    "Sigmoid", "Tanh", "Softmax", "BatchNormalization",
                                    "GlobalAveragePool", "AveragePool", "MaxPool", "GlobalMaxPool",
                                    "Concat", "Slice", "Reshape", "Transpose", "Split", "Gather",
                                    "Pad", "Cast", "Squeeze", "Unsqueeze", "Clip", "Flatten",
                                    "ReduceMax", "ReduceMean" , "Scatter"],
            extra_options={"ActivationSymmetric": True,
                "WeightSymmetric": True,
                "ForceQuantizeNoInputCheck": True}
            )

    # onnnx_model = onnx.load(output_model_path)
    # onnx.checker.check_model(onnnx_model)
    # print(f"✅ Successfully created QDQ model '{output_model_path}'")

def predict(npz_output, ax, color, title_heading):
    try:
        with open('imagenet_class_index.json', 'r') as f:
            class_idx = json.load(f)
        # The keys are strings, so we create a list of class names
        labels = [class_idx[str(k)][1] for k in range(len(class_idx))]
    except FileNotFoundError:
        labels_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        class_idx = requests.get(labels_url).json()
        labels = [class_idx[str(k)][1] for k in range(len(class_idx))]
    
    data = np.load(npz_output)
    output_array = data[data.files[0]].flatten()
    # print("Top-5 Predictions:")
    probabilities = torch.nn.functional.softmax(torch.tensor(output_array), dim=0)
    top5_prob, top5_cat_id = torch.topk(probabilities, 5)
    top5_labels = []
    for i, (idx, val) in enumerate(zip(top5_cat_id, top5_prob)):
        # print(f" {i+1}. {labels[idx]} (Index: {idx}, Probability: {val.item():.4f})")
        top5_labels.append(labels[idx])

    def create_horizontal_barchart(ax, labels, probs, color, title_heading):
        bars = ax.barh(range(5), probs.numpy()[::-1], color=color)
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.02, # x position
                bar.get_y() + bar.get_height()/2.0, # y position
                f'{width:.3f}', # The text
                ha='left', # Horizontal alignment
                va='center' # Vertical alignment
            )
        ax.set_yticks(range(5))
        ax.set_yticklabels([labels[i] for i in probs.argsort(descending=True).numpy()[::-1]])
        ax.set_xlabel("Probability")
        ax.set_title(title_heading)
        ax.invert_yaxis()
    
    create_horizontal_barchart(ax, top5_labels, top5_prob, color, title_heading)

def compare_results(image,
                    tvm_mrvl_fp16_output_path,
                    tvm_mrvl_int8_output_path,
                    llvm_output_fp32_path,
                    llvm_output_int8_path,
                    onnx_output_path,
                    top_k=5):
    """
    Compares the top-k predictions from TVM and ONNX outputs.
    """

    def predict(npz_output):
        try:
            with open('imagenet_class_index.json', 'r') as f:
                class_idx = json.load(f)
            # The keys are strings, so we create a list of class names
            labels = [class_idx[str(k)][1] for k in range(len(class_idx))]
        except FileNotFoundError:
            print("Downloading ImageNet class index...")
            labels_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
            class_idx = requests.get(labels_url).json()
            labels = [class_idx[str(k)][1] for k in range(len(class_idx))]
        
        data = np.load(npz_output)
        output_array = data[data.files[0]].flatten()
        top5_idx = np.argsort(output_array)[::-1][:5]
        top5_vals = output_array[top5_idx]

        print("Top-5 Predictions:")
        import torch
        probs_list = []
        labels = [labels[idx] for idx in top5_idx]
        for i, (idx, val) in enumerate(zip(top5_idx, top5_vals)):
            prob = torch.nn.functional.softmax(torch.tensor(val), dim=0)
            probs_list.append(prob)
            print(f" {i+1}. {labels[idx]} (Index: {idx}, Probability: {prob:.4f})")
        return labels, probs_list
    
    def create_horizontal_barchart(ax, labels, probs, color, title_heading):
        ax.barh(labels, probs[::-1], color=color)
        ax.set_yticks(range(5))
        ax.set_yticklabels(labels[::-1])
        ax.set_xlabel("Probability")
        ax.set_title(title_heading)
        ax.invert_yaxis()

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.subplots_adjust(wspace=0.5, hspace=0.4)

    # First row
    # 1. Input Image
    axs[0, 0].imshow(image)
    axs[0, 0].axis('off')
    axs[0, 0].set_title("Input Image")

    # 2. TVM MLIP fp16
    top5_labels_fp16, top5_probs_fp16 = predict(tvm_mrvl_fp16_output_path)
    create_horizontal_barchart(axs[0, 1], top5_labels_fp16, top5_probs_fp16, 'skyblue', "TVM MLIP fp16")

    # 3. TVM MLIP int8
    top5_labels_int8, top5_probs_int8 = predict(tvm_mrvl_int8_output_path)
    create_horizontal_barchart(axs[0, 2], top5_labels_int8, top5_probs_int8, 'deepskyblue', "TVM MLIP int8")

    # Second row
    # 1. Onnxrt CPU
    top5_labels_onnx, top5_probs_onnx = predict(onnx_output_path)
    create_horizontal_barchart(axs[1, 0], top5_labels_onnx, top5_probs_onnx, 'orange', "Onnxrt CPU")

    # 2. LLVM CPU FP32
    top5_labels_fp32, top5_probs_fp32 = predict(llvm_output_fp32_path)
    create_horizontal_barchart(axs[1, 1], top5_labels_fp32, top5_probs_fp32, 'lightgreen', "LLVM CPU FP32")

    # 3. LLVM CPU int8
    top5_labels_int8_llvm, top5_probs_int8_llvm = predict(llvm_output_int8_path)
    create_horizontal_barchart(axs[1, 2], top5_labels_int8_llvm, top5_probs_int8_llvm, 'lightgreen', "LLVM CPU int8")

    plt.tight_layout()
    plt.show()
