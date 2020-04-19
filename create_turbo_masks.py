from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import glob

image_path = "C:/Users/Srijay/Desktop/Warwick/Datasets/CRAG/Train/Masks/train_1.png"
output_dir = ""
img = image.load_img(image_path)
img_tensor = image.img_to_array(img)
img_tensor = np.clip(img_tensor,0,1)
image.save_img('test.png',img_tensor)