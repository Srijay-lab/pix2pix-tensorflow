import glob
import os
import shutil

def makdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

image_folder = "F:/Datasets/CRAG_LabServer/c2/synthetic_data_provider/1436_500/results/images"

#synthetic_folder = "F:/Datasets/CRAG_LabServer/c2/synthetic_data_provider/1436_500/extracted_synthetic"
real_folder = "F:/Datasets/CRAG_LabServer/c2/synthetic_data_provider/1436_500/extracted_real"

#makdir(synthetic_folder)
makdir(real_folder)

image_paths = glob.glob(os.path.join(image_folder,"*.png"))

l = len(image_paths)

for imagepath in image_paths:
    imname = os.path.split(imagepath)[1]
    if("targets" in imname):
        imname = imname.replace("-targets", "")
        shutil.copy(imagepath, os.path.join(real_folder,imname))
    #if("outputs" in imname):
     #   imname = imname.replace("-outputs","")
      #  shutil.copy(imagepath, os.path.join(synthetic_folder, imname))