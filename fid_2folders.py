# example of calculating the frechet inception distance in Keras
import numpy
import glob
import os
from PIL import Image
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.datasets.mnist import load_data
from skimage.transform import resize

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

folder_1_real = "F:/Datasets/CRAG_LabServer/Test/Grades/1/728_cropped/fid_calculation/images1"
folder_2_real = "F:/Datasets/CRAG_LabServer/Test/Grades/1/728_cropped/fid_calculation/images2"
folder_1_generated = "F:/Datasets/CRAG_LabServer/Test/Grades/1/728_cropped/fid_calculation/run4/1"
folder_2_generated = "F:/Datasets/CRAG_LabServer/Test/Grades/1/728_cropped/fid_calculation/run4/2"
scale = 0
size = 728


paths1_real = glob.glob(os.path.join(folder_1_real,"*.png"))
paths2_real = glob.glob(os.path.join(folder_2_real,"*.png"))
paths1_generated = glob.glob(os.path.join(folder_1_generated,"*.png"))
paths2_generated = glob.glob(os.path.join(folder_2_generated,"*.png"))

def get_real_images(paths):
    images = []
    for path in paths:
        img = Image.open(path)
        img = numpy.asarray(img)
        images.append(img)
    return images

def get_gen_images(paths):
    images = []
    for path in paths:
        img = Image.open(path)
        img = numpy.asarray(img)
        if ('outputs' in path):
            images.append(img)
    return images

images1_real = get_real_images(paths1_real)
images2_real = get_real_images(paths2_real)
images1_gen = get_gen_images(paths1_generated)
images2_gen = get_gen_images(paths2_generated)

images1_real = numpy.array(images1_real)
images1_gen = numpy.array(images1_gen)
images2_real = numpy.array(images2_real)
images2_gen = numpy.array(images2_gen)

# prepare the inception v3 model
model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(size,size,3))

# resize images
if(scale):
    images1_real = scale_images(images1_real, (299,299,3))
    images2_real = scale_images(images2_real, (299,299,3))
    images1_gen = scale_images(images1_gen, (299,299,3))
    images2_gen = scale_images(images2_gen, (299,299,3))

# pre-process images
images1_real = preprocess_input(images1_real)
images2_real = preprocess_input(images2_real)
images1_gen = preprocess_input(images1_gen)
images2_gen = preprocess_input(images2_gen)

fid = calculate_fid(model, images2_real, images1_real)
print('FID between 2 real sets : %.3f' % fid)

fid = calculate_fid(model, images1_real, images1_gen)
print('FID between division 1 real vs gen : %.3f' % fid)

fid = calculate_fid(model, images2_real, images2_gen)
print('FID between division 2 real vs gen : %.3f' % fid)