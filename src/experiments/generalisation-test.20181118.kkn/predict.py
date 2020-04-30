#import keras
#from keras.layers import Input, Dense, merge
#from keras.models import Model
#from keras.layers import Conv2D, MaxPooling2D, Reshape, BatchNormalization
#from keras.layers import Activation, Dropout, Flatten, Dense
#from keras.models import load_model
import numpy as np
from PIL import Image, ImageDraw
import os
import numpy

# Model path
model_path = 'Models/'

def predict(im, y, index, folder = "predictions"):
	lineheight = 20
	padding = 20
	baseh = height-10
	d =  ImageDraw.Draw(im)
	for i in range(5):
		x = width/2 - padding * (i-2)
		d.line([(x, baseh),(x, baseh - lineheight * y[i])], fill=None, width=5)
	im.save(folder + "/" + str(index)+".jpeg")

SOURCE = "AXIO"
SOURCE = "FB"

if(SOURCE == "AXIO"):
	print("Load data")
	X_axio = np.load('Datasets/axionable_data/X_train_axio.npy')
	Y_axio = np.load('Datasets/axionable_data/Y_train_axio.npy')
	Y = np.load("predicted.npy")
	print("Predicted", Y[:10])

	# (26449, 90, 250, 3)
	shape = X_axio.shape
	width = shape[2]
	height = shape[1]

	for index in range(shape[0])[:1000]:
		im = Image.fromarray(X_axio[index])
		y = Y[index]
		predict(im, y, index)

if(SOURCE == "FB"):
	Y = np.load("predicted1.npy")
	print("Predicted", Y[:10])

	path = "D:\\Workspace\\IronCar\\DataSets\\FACE\\car_repo\\records\\"
	files = os.listdir(path)
	index = 0
	for file in files:
		im = numpy.asarray(Image.open(path + file))
		im = im[-110:-20,:,:]

		# (26449, 90, 250, 3)
		shape = im.shape
		width = shape[1]
		height = shape[0]


		im = Image.fromarray(im)
		y = Y[index]
		predict(im, y, index, "predictions1")
		index+=1


#for file in files:
#	im = numpy.asarray(Image.open(path + file))
#	im = im[:90,:,:]
#	im = Image.fromarray(im)
#	y = Y[index]
#	predict(im, y, index)

