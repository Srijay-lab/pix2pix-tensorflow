import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

path = "F:/Datasets/CRAG_LabServer/Test/Grades/2/728_cropped/check_single_big_image/image.png"
output_dir = "F:/Datasets/CRAG_LabServer/Test/Grades/2/728_cropped/check_single_big_image/images"

patchsize = 296
stride = 236
if_pad = 1
new_size = (768,768)

if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def CropImage(image_path):
    image_name = os.path.split(image_path)[1].split('.')[0]
    im = Image.open(image_path)
    size = im.size

    if(if_pad == 0):
        im = im.resize(new_size)

    if(if_pad):
        new_im = Image.new("RGB", new_size)  ## luckily, this is already black!
        new_im.paste(im, ((new_size[0] - size[0]) // 2,
                              (new_size[1] - size[1]) // 2))
        plt.imshow(new_im)
        plt.show()
        print("done")
    else:
        new_im = im

    width, height = new_im.size

    x = 0
    y = 0
    right = 0
    bottom = 0

    while (bottom < height):
        print("Hello")
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
            im_crop = new_im.crop((left, top, right, bottom))
            im_crop_name = image_name + "_" + str(x) + "_" + str(y) + ".png"
            output_path = os.path.join(output_dir, im_crop_name)
            #im = im.resize((256,256))
            im_crop.save(output_path)
            x += stride
        x = 0
        right = 0
        y += stride

#for path in image_paths:

CropImage(path)