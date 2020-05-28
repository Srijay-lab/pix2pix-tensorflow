from PIL import Image
import os, sys
import glob

path = "F:/Datasets/CRAG_LabServer/Test/Grades/1/1024_cropped/images"
outdir = "F:/Datasets/CRAG_LabServer/Test/Grades/1/1024_cropped/resized_728/images/"
resize_len = 728

if not os.path.exists(outdir):
        os.makedirs(outdir)

dirs = os.listdir( path )

image_paths = glob.glob(os.path.join(path,"*.png"))

for path in image_paths:
    imname = os.path.split(path)[1]
    savepath = os.path.join(outdir,imname)
    im = Image.open(path)
    imResize = im.resize((resize_len,resize_len), Image.ANTIALIAS)
    imResize.save(savepath)