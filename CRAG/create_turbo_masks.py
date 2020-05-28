from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import glob
import os

image_dir = "F:/Datasets/CRAG_v2/CRAG/valid/ResizedImages/Masks/"
paths = glob.glob(image_dir+"*.png")
output_dir = "F:/Datasets/CRAG_v2/CRAG/valid/ResizedImages/BMW_Masks"

if not os.path.exists(output_dir):
        os.makedirs(output_dir)
k=1

for path in paths:
    image_name = os.path.split(path)[1]
    img = image.load_img(path)
    img_tensor = image.img_to_array(img)
    #img_tensor = np.clip(img_tensor, 0, 1)
    img_tensor[img_tensor <= 4] = 0
    img_tensor[img_tensor > 4] = 1
    outpath = os.path.join(output_dir,image_name)
    image.save_img(outpath,img_tensor)
    k+=1

print("Number of images done => ",k)