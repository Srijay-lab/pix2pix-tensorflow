import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

folder = "F:/Datasets/CRAG_LabServer/Test/Grades/2/728_cropped/check_single_big_image/results/images"
outpath = "F:/Datasets/CRAG_LabServer/Test/Grades/2/joint"
outfile = os.path.join(outpath,"generated_tile_765.png")

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

image = image/count_masks
#print(image)
#exit(0)
image = image/255.0
print(image.shape)
plt.imshow(image)
plt.show()

matplotlib.image.imsave(outfile, image)