import glob
import os
from PIL import Image
import numpy as np

folder_path = "F:/Datasets/CRAG_LabServer/Train/Grades/3"
masks_input_folder = os.path.join(folder_path, "color_masks")
images_input_folder = os.path.join(folder_path, "images")

output_dir = "F:/Datasets/CRAG_LabServer/Train/Grades/3/728_cropped"
masks_output_folder = os.path.join(output_dir, "masks")
images_output_folder = os.path.join(output_dir, "images")

if not os.path.exists(masks_output_folder):
        os.makedirs(masks_output_folder)
if not os.path.exists(images_output_folder):
        os.makedirs(images_output_folder)

#resize_len = 1500
patch_size = 728
stride = 300

def CropImage(imgname):

    masks_image_path = os.path.join(masks_input_folder,imgname)
    images_image_path = os.path.join(images_input_folder, imgname)
    image_initial = imgname.split('.')[0]
    print(image_initial)

    mask_im = Image.open(masks_image_path)
    image_im = Image.open(images_image_path)
    #Resizing Images
    #mask_im = mask_im.resize((resize_len, resize_len), Image.ANTIALIAS)
    #image_im = image_im.resize((resize_len, resize_len), Image.ANTIALIAS)

    im_np = np.asarray(mask_im)

    #return np.mean(im_np)

    if(np.mean(im_np) <= 50):
        #return
        k=1

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

            im_crop = mask_im.crop((left, top, right, bottom))
            im_crop_name = image_initial + "_" + str(x) + "_" + str(y) + ".png"
            output_path = os.path.join(masks_output_folder, im_crop_name)
            im_crop.save(output_path)

            im_crop = image_im.crop((left, top, right, bottom))
            output_path = os.path.join(images_output_folder, im_crop_name)
            im_crop.save(output_path)

            x += stride

        x = 0
        right = 0
        y += stride

avgs = []

masks_image_paths = glob.glob(os.path.join(masks_input_folder,"*.png"))
image_names = []
for path in masks_image_paths:
    image_names.append(os.path.split(path)[1])
for imgname in image_names:
    #avgs.append(CropImage(imgname))
    CropImage(imgname)

avgs = np.array(avgs)
#print(np.max(avgs))
#print(np.mean(avgs))
