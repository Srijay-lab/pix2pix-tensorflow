import glob
import os
from PIL import Image
import numpy as np
import random

masks_input_folder = "F:/Datasets/CRAG_LabServer/c1/Test/Grades/1/1436_cropped/binary_masks"
images_input_folder = "F:/Datasets/CRAG_LabServer/c1/Test/Grades/1/1436_cropped/results/images"

output_dir = "C:/Users/Srijay/Desktop/Projects/Segmentation/unet/data/crag/SyntheticDataExperiment_sashimi/submitted/test/"
masks_output_folder = os.path.join(output_dir, "masks")
images_output_folder = os.path.join(output_dir, "images")

if not os.path.exists(masks_output_folder):
        os.makedirs(masks_output_folder)
if not os.path.exists(images_output_folder):
        os.makedirs(images_output_folder)

#resize_len = 1024
patch_size = 512
stride = 256
max_background_threshold = 20
#random_set_size = 17

def CropImage(imgname,images_img_name,mbt_index):
    masks_image_path = os.path.join(masks_input_folder,imgname)
    images_image_path = os.path.join(images_input_folder, images_img_name)
    image_initial = images_img_name.split(".")[0] #to give mask same name as image

    mask_im = Image.open(masks_image_path)
    image_im = Image.open(images_image_path)
    #Resizing Images
    #mask_im = mask_im.resize((resize_len, resize_len), Image.ANTIALIAS)
    #image_im = image_im.resize((resize_len, resize_len), Image.ANTIALIAS)

    width, height = mask_im.size

    x = 0
    y = 0
    right = 0
    while (y < height):
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

            mask_crop_mean = np.mean(np.asarray(im_crop))
            save_flag=True
            if (mask_crop_mean <= 12):
                if (mbt_index >= max_background_threshold):
                    save_flag = False
                else:
                    mbt_index = mbt_index + 1
                #print(mbt_index)

            if(save_flag):
                #save mask
                im_crop_name = image_initial + "_" + str(left) + "_" + str(top) + ".png"
                output_path = os.path.join(masks_output_folder, im_crop_name)
                im_crop.save(output_path)

                #Cropping Image and save it
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
#masks_image_paths_random = random.sample(masks_image_paths,random_set_size)

image_names = []
for path in masks_image_paths:
    image_names.append(os.path.split(path)[1])
for imgname in image_names:
    target_img_name = imgname.split(".")[0] + "-targets.png"
    generated_img_name = imgname.split(".")[0] + "-outputs.png"
    mbt_index = CropImage(imgname,target_img_name,mbt_index)
    mbt_index = CropImage(imgname,generated_img_name,mbt_index)

avgs = np.array(avgs)
#print(np.max(avgs))
#print(np.mean(avgs))
