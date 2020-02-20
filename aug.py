import os
import cv2
import numpy as np 
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img


aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.10,
	horizontal_flip=True,
	fill_mode="nearest")

imgs = os.listdir('/home/vs/Desktop/pan images/rawimage/')

#print(imgs)
width = 512
height = 300
dim = (width, height)

for img in imgs:
	image = load_img('/home/vs/Desktop/pan images/rawimage/'+img)
	image = img_to_array(image)
	#resize image
	image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

	
	image = np.expand_dims(image, axis=0)
	i = 0
	imageGen = aug.flow(image, batch_size=1, save_to_dir='/home/vs/Desktop/pan images/IMGA/', save_prefix="image", save_format="jpg")
	for batch in imageGen:
		i+=1
		if i>5:
			break


