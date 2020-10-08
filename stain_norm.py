import glob
import os

tiatoolbox_loc = "C:/Users/Srijay/Desktop/Projects/tiatoolbox_private/"

input_image_dir = 'F:/Datasets/DigestPath/safron/test/images'

target_image = 'F:/Datasets/DigestPath/safron/train/stain.jpg'

output_dir = 'F:/Datasets/DigestPath/safron/test/stain_images'

image_paths = glob.glob(input_image_dir + "/*.jpg")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read Images
for image_path in image_paths:
    cmd = "python " + tiatoolbox_loc + "tiatoolbox.py stain_normalise --path " + image_path + " --target_image_path " + target_image + " --save_path " + output_dir
    try:
        os.system(cmd)
    except:
        print("Some error occurred")