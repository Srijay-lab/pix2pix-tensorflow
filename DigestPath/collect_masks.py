import numpy as np
import matplotlib
import os
import glob
from tensorflow.keras.preprocessing import image
import PIL

PIL.Image.MAX_IMAGE_PIXELS = 933120000

masks_folder = "F:/Datasets/DigestPath/masks"
images_folder = "F:/Datasets/DigestPath/images"
outfolder = "F:/Datasets/DigestPath/tri_masks"

masks_paths = glob.glob(os.path.join(masks_folder,"*.png"))

if not os.path.exists(outfolder):
        os.makedirs(outfolder)

def extract_image(mask_path):
    mask_name = os.path.split(mask_path)[1]
    mask_name = mask_name.split(".")[0]+".png"
    img_name = mask_name.split(".")[0]+".jpg"
    img_path = os.path.join(images_folder,img_name)
    mask = image.load_img(mask_path)
    img = image.load_img(img_path)
    mask_np = image.img_to_array(mask)
    image_np = image.img_to_array(img)
    if mask_np.shape[2] == 4:
        mask_np = mask_np[:,:,:3]
    if image_np.shape[2] == 4:
        image_np = image_np[:,:,:3]
    w, h, d = image_np.shape

    new_mk = np.empty([w, h, d])
    mask_np[mask_np<128] = 0
    mask_np[mask_np>=128] = 255

    for i in range(0,w):
        for j in range(0,h):
            if(np.all((mask_np[i][j]==0))):
                if(np.mean(image_np[i][j]) > 230): #background
                    new_mk[i][j] = [0,0,0]
                else:
                    new_mk[i][j] = [255, 0, 0] #stroma
            else:
                new_mk[i][j] = [255, 255, 255] #tumor
    print(mask_name)
    image.save_img(os.path.join(outfolder,mask_name),new_mk)

for path in masks_paths:
    extract_image(path)