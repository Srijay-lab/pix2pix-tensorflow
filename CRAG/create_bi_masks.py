import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob

folder_path = "F:/Datasets/CRAG_LabServer/Test/Grades/1/1436_cropped/SyntheticCohort/results/images"
masks_input_folder = os.path.join(folder_path, "color_masks")
images_input_folder = os.path.join(folder_path, "images")
outdir = "F:/Datasets/CRAG_LabServer/Test/Grades/1/Two_Parallel_AFROS/glands"
IsGlandMask = True #Make true if you want gland vs everything as background and False if stroma vs everything as background

outdir_masks = outdir + "/masks"
outdir_images = outdir + "/images"

if not os.path.exists(outdir_masks):
        os.makedirs(outdir_masks)

if not os.path.exists(outdir_images):
        os.makedirs(outdir_images)

def GenerateBiMask(imgname):
    masks_image_path = os.path.join(masks_input_folder, imgname)
    images_image_path = os.path.join(images_input_folder, imgname)
    mk = Image.open(masks_image_path)
    im = Image.open(images_image_path)
    im_np = np.asarray(im)
    mk_np = np.asarray(mk)
    #exit(0)
    # remove alpha channel
    if mk_np.shape[2] == 4:
        mk_np = mk_np[:, :, :3]

    w,h,d = im_np.shape

    new_mk = np.empty([w,h,d])
    new_im = np.empty([w,h,d])

    for i in range(0,w):
        for j in range(0,h):
            if(mk_np[i][j][1] == 255): #gland
                new_mk[i][j] = [0,255,0]
                new_im[i][j] = im_np[i][j]
            else: #background
                new_mk[i][j] = [0,0,255]
                new_im[i][j] = [0,0,0]

    new_mk = new_mk / 255.0
    new_im = new_im / 255.0

    outpath_mask = os.path.join(outdir_masks,imgname)
    matplotlib.image.imsave(outpath_mask, new_mk)
    outpath_image = os.path.join(outdir_images, imgname)
    matplotlib.image.imsave(outpath_image, new_im)

masks_image_paths = glob.glob(os.path.join(masks_input_folder,"*.png"))
image_names = []
for path in masks_image_paths:
    image_names.append(os.path.split(path)[1])

for imgname in image_names:
    GenerateBiMask(imgname)



