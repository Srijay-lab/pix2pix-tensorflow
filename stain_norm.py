import glob
import os

tiatoolbox_loc = "C:/Users/Srijay/Desktop/Projects/tiatoolbox_private/"

input_image_dir = 'F:/Datasets/CRAG_LabServer/Train/Grades/1/1024_cropped/images'

target_image = 'F:/Datasets/CRAG_LabServer/Train/Grades/1/1024_cropped/stain.png'

output_dir = 'F:/Datasets/CRAG_LabServer/Train/Grades/1/1024_cropped/stain_images'

image_paths = glob.glob(input_image_dir + "/*.png")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read Images
for image_path in image_paths:
    cmd = "python " + tiatoolbox_loc + "tiatoolbox.py stain_normalise --path " + image_path + " --target_image_path " + target_image + " --save_path " + output_dir
    try:
        os.system(cmd)
    except:
        print("Some error occurred")