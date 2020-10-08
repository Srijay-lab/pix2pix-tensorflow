import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image

size = 728
num_images = 1
gland_image_path = "F:/Datasets/CRAG_LabServer/SyntheticCohort/wsi/gland1.png"
outpath = "F:/Datasets/CRAG_LabServer/SyntheticCohort/wsi/"

for k in range(0,num_images):
    gland_mask = Image.open(gland_image_path)
    gland_mask = np.asarray(gland_mask)
    prob_count=1

    if gland_mask.shape[2] == 4:
        gland_mask = gland_mask[:,:,:3]

    stromal_mask = np.random.randint(10, size=(size,size))

    random_mask = np.empty([size,size,3])

    random_mask[:, :, 0][stromal_mask[:, :]<=prob_count] = 0 #white background : Blue color
    random_mask[:, :, 1][stromal_mask[:, :]<=prob_count] = 0 #white background : Blue color
    random_mask[:, :, 2][stromal_mask[:, :]<=prob_count] = 255 #white background : Blue color

    random_mask[:, :, 0][stromal_mask[:, :] > prob_count] = 255  # stroma : Red color
    random_mask[:, :, 1][stromal_mask[:, :] > prob_count] = 0  # stroma : Red color
    random_mask[:, :, 2][stromal_mask[:, :] > prob_count] = 0  # stroma : Red color

    random_mask[:, :, 0][gland_mask[:, :, 0] == 128] = 0  # gland : Green color
    random_mask[:, :, 1][gland_mask[:, :, 1] == 128] = 255  # gland : Green color
    random_mask[:, :, 2][gland_mask[:, :, 2] == 128] = 0  # gland : Green color

    print("Done")
    random_mask = random_mask/255.0
    imname = "synthetic_"+str(k)+".png"
    matplotlib.image.imsave(os.path.join(outpath,imname), random_mask)