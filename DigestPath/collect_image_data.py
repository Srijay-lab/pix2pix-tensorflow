import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.preprocessing import image
import os
import glob
import PIL

#PIL.Image.MAX_IMAGE_PIXELS = 933120000
images = "F:/Datasets/DigestPath/mask_npy"
outfolder = "F:/Datasets/DigestPath/masks"

paths = glob.glob(os.path.join(images,"*.npy"))

if not os.path.exists(outfolder):
        os.makedirs(outfolder)
ws = []
hs = []

def extract_image(path):
    imname = os.path.split(path)[1]
    imname = imname.split(".")[0]+".png"
    image = np.load(path)
    print(image.shape)
    w,h = image.shape
    matplotlib.image.imsave(os.path.join(outfolder,imname), image)
    ws.append(w)
    hs.append(h)

for path in paths:
    if("neg" in path):
        print(path)
        extract_image(path)