import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
import shutil
import matplotlib.cm as cm

folder_path = "F:/Datasets/DigestPath/safron/test/3/964_800/results_tmi3_100/images"

outdir = "F:/Datasets/DigestPath/safron/test/3/964_800/unet_train_data"

outdir_masks = outdir + "/binary_masks_syn"
outdir_images = outdir + "/synthetic_images"

if not os.path.exists(outdir_masks):
        os.makedirs(outdir_masks)

if not os.path.exists(outdir_images):
        os.makedirs(outdir_images)


def GenerateBiMask(mask_name,img_name):
    mask_path = os.path.join(folder_path, mask_name)
    image_path = os.path.join(folder_path, img_name)

    mk = Image.open(mask_path)
    mk_np = np.asarray(mk)

    im = Image.open(image_path)
    im_np = np.asarray(im)

    # remove alpha channel
    if mk_np.shape[2] == 4:
        mk_np = mk_np[:, :, :3]

    w,h,d = im_np.shape

    new_mk = np.empty([w,h])

    for i in range(0,w):
        for j in range(0,h):
            if(mk_np[i][j][1] == 255): #gland
                new_mk[i][j] = 255
            else: #background
                new_mk[i][j] = 0
    print("done")
    if (np.mean(new_mk) >= 10):  # white portion
        new_mk = new_mk / 255.0

        outpath_mask = os.path.join(outdir_masks,img_name)
        matplotlib.image.imsave(outpath_mask, new_mk, cmap=cm.gray)

        shutil.copy(image_path,os.path.join(outdir_images,img_name+".png"))


image_paths = glob.glob(os.path.join(folder_path,"*.png"))

image_names = []
imt_dict = {}
for path in image_paths:
    imname = os.path.split(path)[1]
    imname = imname.split("-")[0]
    imno = imname.split("_")[1]
    if(imno in imt_dict):
        imt_dict[imno]+=1
    else:
        imt_dict[imno]=1
    if(imt_dict[imno]<=30):
        if(imname not in image_names):
            image_names.append(imname)

for imgname in image_names:
    mask = imgname + "-inputs.png"
    img = imgname + "-outputs.png"
    GenerateBiMask(mask,img)



