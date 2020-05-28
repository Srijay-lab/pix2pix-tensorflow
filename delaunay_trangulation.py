import cv2
import matplotlib.pyplot as plt

path = "C:/Users/Srijay/Desktop/Warwick/Datasets/PanNuke"

img = cv2.imread(path)
img = img/255.0
plt.imshow(img)
plt.show()