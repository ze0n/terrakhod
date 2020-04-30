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

#For Keras
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, save_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


from tensorflow.python.keras.layers import Input, Dense, Flatten, Dropout, Conv2D
from tensorflow.python.keras.models import Model
import re

def model_categorical(input_size=  (90, 250, 3), dropout=0.1):
    '''Generate an NVIDIA AutoPilot architecture.
    Input_size: Image shape (90, 250, 3), adjust to your desired input.
    Dropout: Proportion of dropout used to avoid model overfitting.
    This model ONLY predicts steering angle as a 5-elements array encoded with a Softmax output.
    The model is already compiled and ready to be trained.
    '''

    
    img_in = Input(shape=input_size, name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation='relu')(img_in)       # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)                                    # Classify the data into 100 features, make all negatives 0
    x = Dropout(dropout)(x)                                                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)                                     # Classify the data into 50 features, make all negatives 0
    x = Dropout(dropout)(x)                                                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    
    #categorical output of the angle
    angle_out = Dense(5, activation='softmax', name='angle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0
    
    model = Model(inputs=[img_in], outputs=[angle_out])
    
    return model

def load_trained_model(weights_path):
   model = model_categorical()
   model.load_weights(weights_path)
   return model

print("Load model")
#model = load_model('final_model.test1.hdf5')
#model = load_model('./final_model_weights.hdf5', custom_objects=[])
model = load_trained_model('./final_model_weights.hdf5')
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
	Y1 = []
	for file in files:
		g = re.match("frame_\d+_gas_[\d\.\+\-]+_dir_([\d\.\+\-]+)\.jpg", file)
		y = float(g.groups()[0])
		im = numpy.asarray(Image.open(path + file))
		x = im[-110:-20,:,:]
		X.append(x)
		Y1.append(y)

	Y1 = np.array(Y1)
	X = np.array(X)

	Y = model.predict(X)
	print("Predicted", Y[0])
	np.save("predicted1.npy", Y)
	np.save("predicted1_valid.npy", Y1)
