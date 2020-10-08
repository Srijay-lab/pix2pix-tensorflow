import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

input_dir = "F:/Datasets/DigestPath/safron/test/3/1436_1200/results_tmi3/images"
output_dir = "C:/Users/Srijay/Desktop/Projects/Segmentation/unet/data/crag/TMI_EXP_2/setup3/train/crag_train/real/crag_real_data_provider"

patchsize = 964
stride = 600

if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def CropImage(image_path):

    image_name = os.path.split(image_path)[1].split('.')[0]
    im = Image.open(image_path)
    width, height = im.size

    x = 0
    y = 0
    right = 0
    bottom = 0

    while (bottom < height):
        while (right < width):
            left = x
            top = y
            right = left + patchsize
            bottom = top + patchsize
            if (right > width):
                offset = right - width
                right -= offset
                left -= offset
            if (bottom > height):
                offset = bottom - height
                bottom -= offset
                top -= offset
            im_crop = im.crop((left, top, right, bottom))
            im_crop_name = image_name + "_" + str(left) + "_" + str(top) + ".png"
            output_path = os.path.join(output_dir, im_crop_name)
            im_crop.save(output_path)
            x += stride
        x = 0
        right = 0
        y += stride

image_paths = glob.glob(os.path.join(input_dir,"*.png"))
for path in image_paths:
    CropImage(path)