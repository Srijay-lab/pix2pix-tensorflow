import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import imageio
import os

maskspath = "F:/Datasets/PanNuke/Fold1/masks_npy/fold1/masks.npy"
outfolder = "F:/Datasets/PanNuke/Fold1/bw_masks/"
typespath = "F:/Datasets/PanNuke/Fold1/images_npy/fold1/types.npy"
types = np.load(typespath)

if not os.path.exists(outfolder):
        os.makedirs(outfolder)

color_dict = {
                0 : [0,0,0],
                1 : [0,128,0],
                2 : [255,255,0],
                3 : [255,0,0],
                4 : [0,0,255],
                5 : [255,255,255]
              }

def CreateMaskImage(mask,type,index):
    k = np.zeros((256, 256, 3))
    l = 256
    for i in range(0,l):
        for j in range(0,l):
            #k[i][j] = color_dict[np.argmax(mask[i][j])]
            if(np.argmax(mask[i][j]) == 5):
                k[i][j] = [0,0,0]
            else:
                k[i][j] = [255,255,255]
    imagename = type + "_" + str(index) + ".png"
    imagepath = os.path.join(outfolder, imagename)
    matplotlib.image.imsave(imagepath, k)

masks = np.load(maskspath)
total = len(masks)
for i in range(0,total):
    mask = masks[i]
    type = types[i]
    CreateMaskImage(mask,type,i)