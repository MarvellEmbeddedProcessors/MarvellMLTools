# SPDX-License-Identifier: Apache-2.0

# Based on imagenet_preprocess.py from github.com/onnx/models repo
# https://github.com/onnx/models/blob/main/validated/vision/classification/imagenet_preprocess.py

import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image_file", type=str, required=True)
parser.add_argument("--bin_file", type=str, required=True)

# resize so that the shorter side is 256, maintaining aspect ratio
def image_resize(image, min_len):
        ratio = float(min_len) / min(image.size[0], image.size[1])
        if image.size[0] > image.size[1]:
            new_size = (int(round(ratio * image.size[0])), min_len)
        else:
            new_size = (min_len, int(round(ratio * image.size[1])))
        image = image.resize(new_size, Image.BILINEAR)
        return np.array(image)

# Crop centered window 224x224
def crop_center(image, crop_w, crop_h):
        h, w, c = image.shape
        start_x = w // 2 - crop_w // 2
        start_y = h // 2 - crop_h // 2
        return image[start_y:start_y + crop_h, start_x:start_x + crop_w, :]

def preprocess_image(image):
    # resize and crop
    image = image_resize(image, 256)
    image = crop_center(image, 224, 224)

    # transpose
    image = image.transpose(2, 0, 1)

    # convert the input data into the float32 input
    img_data = image.astype('float32')

    # normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]

    # add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

if __name__ == '__main__':
    args = parser.parse_args()
    with Image.open(args.image_file) as img_file:
        # preprocessing image
        image = preprocess_image(img_file)

        bin_data = image.tobytes()
        with open(args.bin_file, 'wb') as bin_file:
            bin_file.write(bin_data)
        print(f"successfully converted {args.image_file} to {args.bin_file}")
