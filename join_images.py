import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

folder = "F:/Datasets/CRAG_LabServer/c1/Test/Grades/1/728_cropped/single/results_9d1g/images"
outpath = "F:/Datasets/CRAG_LabServer/c1/Test/Grades/1/728_cropped/single/"
outfile = os.path.join(outpath,"tissue_mask_synthetic-tmi-single-2.png")

if not os.path.exists(outpath):
        os.makedirs(outpath)

paths = glob.glob(os.path.join(folder,"*.png"))

hight = 728
width = 728
patch = 256

image = np.zeros((hight,width,3))
count_masks = np.zeros((hight,width,3))
k=0
for path in paths:
    if('outputs' in path):
        imname = os.path.split(path)[1].split("-")[0]
        imname = imname.split("_")
        y,x = int(imname[-2]),int(imname[-1])
        img = Image.open(path)
        img = np.asarray(img)
        #print("X => ",x," Y => ",y)
        image[x:x+patch,y:y+patch,:] += img
        count_masks[x:x+patch,y:y+patch,:]+=1.0
        k+=1

count_masks = count_masks.clip(min=1)
#count_masks[count_masks>1]=1000
#image[image>255]=0
image = image/count_masks
#print(image)
#exit(0)
image = image/255.0
print(image.shape)
#plt.imshow(image)
#plt.show()
print("Done")
matplotlib.image.imsave(outfile, image)