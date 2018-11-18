import keras
from keras.layers import Input, Dense, merge
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
import numpy as np
from PIL import Image, ImageDraw
import os
import numpy

print("Load model")
model = load_model('final_model.test1.hdf5')
print("OK")
model.summary()

SOURCE = "AXIO"
SOURCE = "FB"


if(SOURCE == "AXIO"):
	print("Load data")
	X_axio = np.load('Datasets/axionable_data/X_train_axio.npy')
	Y_axio = np.load('Datasets/axionable_data/Y_train_axio.npy')
	Y = model.predict(X_axio)
	print("Predicted", Y[0])
	np.save("predicted.npy", Y)

if(SOURCE == "FB"):
	path = "D:\\Workspace\\IronCar\\DataSets\\FACE\\car_repo\\records\\"
	files = os.listdir(path)
	X = []
	for file in files:
		im = numpy.asarray(Image.open(path + file))
		x = im[-110:-20,:,:]
		X.append(x)

	X = np.array(X)

	Y = model.predict(X)
	print("Predicted", Y[0])
	np.save("predicted1.npy", Y)
