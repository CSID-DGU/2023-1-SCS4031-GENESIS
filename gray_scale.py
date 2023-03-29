import numpy as np
from os import listdir
from random import *
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image

path = 'content/images/val'
newpath = 'content/grayscale_images/val'

for file in listdir(path):
   img = Image.open(path+file).convert('L')
   img_numpy = np.array(img,'uint8')
   cv2.imwrite(newpath+file,img_numpy)