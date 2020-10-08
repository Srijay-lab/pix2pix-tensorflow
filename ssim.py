# example of calculating the frechet inception distance in Keras
import numpy
import glob
import os
from skimage.measure import compare_ssim
from PIL import Image
import cv2
import numpy as np

# calculate SSIM
def calculate_average_ssim(images1, images2):
    ssim_sum=0
    n = len(images1)
    ssims_list = []
    for i in range(0,n):
        (ssim,diff) = compare_ssim(images1[i], images2[i], gaussian_weights=True, full=True, sigma=1.5, use_sample_covariance=False, multichannel=True)
        ssims_list.append(ssim)
    ssims_list = np.array(ssims_list)
    mean,std = np.mean(ssims_list),np.std(ssims_list)
    return mean,std

folder = "F:/Datasets/CRAG_LabServer/Test/Grades/1/1200_cropped/results_run4/images"

paths = glob.glob(os.path.join(folder,"*.png"))

original_images = []
generated_images = []
image_names = []

for path in paths:
    if('outputs' in path):
        imname = os.path.split(path)[1].split("-")
        imname = "-".join([imname[0],imname[1]])
        image_names.append(imname)
#image_names = ["H09-16145_A2H_E_1_1_grade_1_14_0_500"]
print(image_names)
#exit(0)
for imname in image_names:
    or_imname = imname+"-targets.png"
    gn_imname = imname+"-outputs.png"

    #or_img = Image.open(os.path.join(folder,or_imname))
    #or_img = numpy.asarray(or_img)
    or_img = cv2.imread(os.path.join(folder,or_imname))
    or_img = cv2.cvtColor(or_img, cv2.COLOR_BGR2RGB)
    original_images.append(or_img)

    #gn_img = Image.open(os.path.join(folder, gn_imname))
    #gn_img = numpy.asarray(gn_img)
    gn_img = cv2.imread(os.path.join(folder, gn_imname))
    gn_img = cv2.cvtColor(gn_img, cv2.COLOR_BGR2RGB)
    generated_images.append(gn_img)

print(len(original_images))
print(len(generated_images))

# fid between images1 and images1
ssim_avg,ssim_std = calculate_average_ssim(original_images, generated_images)
print("Average SSIM => ",ssim_avg)
print("STD SSIM => ",ssim_std)