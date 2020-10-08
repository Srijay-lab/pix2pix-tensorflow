import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib
import os
import glob
from PIL import Image

size = 1000
num_images = 30
outpath = "F:/Datasets/TheCOT/ConstructedFromSoftware/tmi"
get_from_thecot = True
thecot_path = "F:/Datasets/TheCOT/ConstructedFromSoftware/exp"

if not os.path.exists(outpath):
    os.makedirs(outpath)

if(get_from_thecot):
    gland_masks = []
    thecot_image_names = []
    gland_image_paths = glob.glob(os.path.join(thecot_path, "*.png"))
    for path in gland_image_paths:
        if("crypts" in path):
            image_name = os.path.split(path)[1]
            thecot_image_names.append(image_name)
            img = Image.open(path).convert('L')
            gland_mask = np.asarray(img)
            gland_masks.append(gland_mask)
    num_images = len(gland_masks)

for k in range(0,num_images):
    if(get_from_thecot):
        gland_mask = gland_masks[k]
        print(gland_mask.shape)
        gland_pixel_val = 255


    random_mask = np.empty([893,974,3])

    w,h,d = random_mask.shape

    for i in range(0,w):
        for j in range(0,h):
            if(gland_mask[i][j]==gland_pixel_val):
                random_mask[i][j] = [0, 255, 0] #gland
            elif(random.randint(0, 10)==0):
                random_mask[i][j] = [0,0,255] #blue background
            else:
                random_mask[i][j] = [255,0,0] #stroma

    random_mask = random_mask/255.0
    if(get_from_thecot):
        imname = thecot_image_names[k]
    else:
        imname = "synthetic_"+str(k+1)
    print(imname)
    matplotlib.image.imsave(os.path.join(outpath,imname), random_mask)