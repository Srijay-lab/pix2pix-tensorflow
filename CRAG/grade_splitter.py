import glob
import os
import shutil

def makdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

image_folder = "F:/Datasets/CRAG_LabServer/Test/images/"
mask_folder = "F:/Datasets/CRAG_LabServer/Test/masks/"
output_folder = "F:/Datasets/CRAG_LabServer/Test/Grades"

image_paths = glob.glob(os.path.join(image_folder,"*.png"))
mask_paths = glob.glob(os.path.join(mask_folder,"*.png"))
l = len(image_paths)

for i in range(0,l):
    imagepath = image_paths[i]
    maskpath = mask_paths[i]
    image_name = os.path.split(imagepath)[1].split('_')
    grade = image_name[-2]
    imoutdir = os.path.join(output_folder,grade)
    imoutdir = os.path.join(imoutdir,"images")
    makdir(imoutdir)
    mskoutdir = os.path.join(output_folder,grade)
    mskoutdir = os.path.join(mskoutdir, "masks")
    makdir(mskoutdir)
    shutil.copy(imagepath,imoutdir)
    shutil.copy(maskpath,mskoutdir)