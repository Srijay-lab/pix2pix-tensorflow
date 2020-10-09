import glob
import os
import shutil

def makdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

folder = "F:/Datasets/CRAG_LabServer/Test/Grades/1/cropped/images"
output_folder = "F:/Datasets/CRAG_LabServer/Test/Grades/1/cropped/test"

makdir(output_folder)

image_paths = glob.glob(os.path.join(folder,"*.png"))
l = len(image_paths)

for path in image_paths:
    image_name = os.path.split(path)[1]
    if("H09-00622_A2H_E_1_3_grade_1_12" in image_name):
        shutil.copy(path,output_folder)