import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

path = "F:/Datasets/CRAG_LabServer/c1/Test/Grades/1/1436_cropped/results_1/run4"
output_dir = "F:/Datasets/CRAG_LabServer/c1/Test/Grades/1/1436_cropped/results_1/run4_glued_single"

patchsize = 256
stride = 236

if not os.path.exists(output_dir):
        os.makedirs(output_dir)

image_paths = glob.glob(os.path.join(path,"*.png"))

def ExtractGluePatches(image_path):
    image_name = os.path.split(image_path)[1].split('.')[0]
    print(image_name)
    im = Image.open(image_path)

    new_im = im

    width, height = new_im.size

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
            glue_x = left + stride - 28
            glue_x_end = left + patchsize + 28
            save = 1
            if(glue_x_end>width):
                save=0
            glue_y = top
            glue_y_end = top + 76
            if (glue_y_end > height):
                save=0
            if(save==1):
                im_crop = new_im.crop((glue_x, glue_y, glue_x_end, glue_y_end))
                im_crop_name = image_name + "_" + str(glue_x) + "_" + str(glue_y) + ".png"

                output_path = os.path.join(output_dir, im_crop_name)
                im_crop.save(output_path)
            x += stride
        x = 0
        right = 0
        y += stride

for path in image_paths:
    if("outputs" in path or "targets" in path):
        if("H09-00107_A1H_E_1_2_grade_1_2_500_0" in path):
            ExtractGluePatches(path)