import glob
import os
from PIL import Image
import numpy as np

patch_size = 1436
stride = 200
folder_path = "F:/Datasets/CRAG_LabServer/c3/test"
masks_input_folder = os.path.join(folder_path, "binary_masks")
images_input_folder = os.path.join(folder_path, "images")

output_dir = "F:/Datasets/CRAG_LabServer/c3/test/1436_cropped"
masks_output_folder = os.path.join(output_dir, "binary_masks")
images_output_folder = os.path.join(output_dir, "binary_images")

if not os.path.exists(masks_output_folder):
        os.makedirs(masks_output_folder)
if not os.path.exists(images_output_folder):
        os.makedirs(images_output_folder)

def CropImage(imgname,mbt_index):

    masks_image_path = os.path.join(masks_input_folder,imgname)
    image_initial = imgname.split('.')[0]
    images_image_path = os.path.join(images_input_folder, image_initial+".png")

    mask_im = Image.open(masks_image_path).convert('RGB')
    print(imgname)

    image_im = Image.open(images_image_path).convert('RGB')

    width, height = mask_im.size

    x = 0
    y = 0
    right = 0
    bottom = 0

    while (bottom < height):
        while (right < width):
            left = x
            top = y
            right = left + patch_size
            bottom = top + patch_size
            if (right > width):
                offset = right - width
                right -= offset
                left -= offset
            if (bottom > height):
                offset = bottom - height
                bottom -= offset
                top -= offset

            #Cropping Mask
            im_crop = mask_im.crop((left, top, right, bottom))

            if(np.mean(np.asarray(im_crop))>50):
                im_crop_name = image_initial + "_" + str(left) + "_" + str(top) + ".png"
                output_path = os.path.join(masks_output_folder, im_crop_name)
                im_crop.save(output_path)

                #Cropping Image
                im_crop = image_im.crop((left, top, right, bottom))
                output_path = os.path.join(images_output_folder, im_crop_name)
                im_crop.save(output_path)

            x += stride

        x = 0
        right = 0
        y += stride

    return mbt_index

avgs = []
mbt_index = 0
masks_image_paths = glob.glob(os.path.join(masks_input_folder,"*.png"))
image_names = []
for path in masks_image_paths:
    print(path)
    image_names.append(os.path.split(path)[1])
for imgname1 in image_names:
    mbt_index = CropImage(imgname1,mbt_index)

avgs = np.array(avgs)

