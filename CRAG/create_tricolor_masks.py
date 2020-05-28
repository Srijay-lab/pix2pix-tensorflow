import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob

folder_path = "F:/Datasets/CRAG_LabServer/Train/Grades/3"
masks_input_folder = os.path.join(folder_path, "masks")
images_input_folder = os.path.join(folder_path, "images")
outdir = "F:/Datasets/CRAG_LabServer/Train/Grades/3/color_masks"

if not os.path.exists(outdir):
        os.makedirs(outdir)

def GenerateTriMask(imgname):
    masks_image_path = os.path.join(masks_input_folder, imgname)
    images_image_path = os.path.join(images_input_folder, imgname)
    mk = Image.open(masks_image_path)
    im = Image.open(images_image_path)
    im_np = np.asarray(im)
    mk_np = np.asarray(mk)

    w,h,d = im_np.shape

    new_mk = np.empty([w,h,d])

    for i in range(0,w):
        for j in range(0,h):
            if(mk_np[i][j] > 230): #gland
                new_mk[i][j] = [0,255,0]
            elif(np.mean(im_np[i][j]) > 230): #background
                new_mk[i][j] = [0,0,255]
            else: #tissue
                new_mk[i][j] = [255,0,0]
    #new_mk = new_mk / 255.0
    outpath = os.path.join(outdir,imgname)
    matplotlib.image.imsave(outpath, new_mk)

masks_image_paths = glob.glob(os.path.join(masks_input_folder,"*.png"))
image_names = []
for path in masks_image_paths:
    image_names.append(os.path.split(path)[1])

print(image_names)

for imgname in image_names:
    GenerateTriMask(imgname)



