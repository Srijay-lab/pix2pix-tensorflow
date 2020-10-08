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
from PIL import Image
import random

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        img = Image.fromarray(image)
        img = img.resize(size=new_shape)
        img = numpy.asarray(img)
        # store
        images_list.append(img)
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

folder = "F:/Datasets/DigestPath/safron/test/3/1436_1200/results_tmi3/images"

paths = glob.glob(os.path.join(folder,"*.png"))
#paths = random.sample(paths,1000)

size = 1436
scale_299 = 0

target_images = []
output_images = []

for path in paths:
    img = Image.open(path)
    img = numpy.asarray(img)
    #if(numpy.mean(img)>230):
        #continue
    if('outputs' in path):
        output_images.append(img)
    elif('targets' in path):
        target_images.append(img)

target_images = numpy.array(target_images)
output_images = numpy.array(output_images)
num_images = len(output_images)
print(num_images)
print(len(target_images))

# resize images
if(scale_299):
    target_images = scale_images(target_images, (299,299))
    output_images = scale_images(output_images, (299,299))
    size = 299

# prepare the inception v3 model
model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(size,size,3))

images_random = randint(0, 255, num_images*size*size*3)
images_random = images_random.reshape((num_images,size,size,3))
images_random = images_random.astype('float32')

#target_images = target_images.astype('float32')
#output_images = output_images.astype('float32')

# pre-process images
target_images = preprocess_input(target_images)
output_images = preprocess_input(output_images)
images_random = preprocess_input(images_random)

# fid between images1 and images1
fid = calculate_fid(model, target_images, target_images)
print('FID (same): %.3f' % fid)
# fid between images1 and random
fid = calculate_fid(model, target_images, images_random)
print('FID (random): %.3f' % fid)
# fid between images1 and images2
fid = calculate_fid(model, target_images, output_images)
print('FID (Predicted): %.3f' % fid)