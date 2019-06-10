
PATH = 'terrakhod-20181124/1/'

import numpy as np

def steering_encoder(steering_str):
    '''
    Converts the steering with appears in the file names to [0,0,1] (left),
    [0,1,0] (straight), or [1,0,0] (right)
    '''
    steering_float = float(steering_str)
    if steering_float == 0:
        return [0,1,0]
    elif steering_float < 0:
        return [0,0,1]
    else:
        return [1,0,0]


import re

def filename_decoder(file_name):
    '''exemple: frame_390_gas_-0.82_dir_-1.0.jpg
    '''
    gas, steering_str = re.search(r'_gas_(.+?)_dir_(.+?)\.jpg', file_name).groups()
    steering = steering_encoder(steering_str)
    return gas, steering

from os import listdir
from os.path import isfile, join

def list_files_in_dir(my_path):
    files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
    return files

files = list_files_in_dir(PATH )

import skimage
from skimage.io import imread, imshow

def read_image_and_steering(file_name):

    image = skimage.io.imread(PATH + file_name)
    gas, steer = filename_decoder(file_name)
    return image, steer 


def horizontal_flip(img, steer):
    """Horizontal image flipping and angle correction.
    Img: Input image to transform in Numpy array.
    Angle: Corresponding label. Must be a 5-elements Numpy Array.
    """    
    return np.fliplr(img), np.flipud(steer)


import cv2


def augment_brightness_camera_images(image):
    '''Random bright augmentation (both darker and brighter).
    
    Returns:
    Transformed image and label.
    '''
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def add_random_shadow(image):
    '''Add random dark shadows to a given image.
    Returns:
    Transformed image and label.
    '''
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

import skimage.exposure as exposure
import random

def night_effect(img, vmin=185, vmax=195):
    """Change road color to black simulating night road.
    Returns
    Transformed image and label.
    """
    limit = random.uniform(vmin,vmax)
    low_limit = 146 
    int_img = exposure.rescale_intensity(img, in_range=(low_limit,limit), out_range='dtype')
    
    return int_img

def adjust_gamma_dark(image, min_=0.7, max_=0.8):
    '''Gamma correction to generate darker images.
    Image: Image in Numpy format (90,250,3)
    Label: Corresponding label of the image.
    Min: Minimum gamma value (the lower the darker)
    Max: Maximum gamma value (the higher the brigther) 
    Return:
    Transformed image and label.
    '''
    # build a lookup table mapping the pixel values [0, 255] to
    gamma = random.uniform(min_,max_)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def augment(image, steer):
    
    if np.random.random() > 0.75:
        image = adjust_gamma_dark(image)
    elif np.random.random() > 0.75:
        image = night_effect(image)
    elif np.random.random() > 0.75:
        image = add_random_shadow(image)
    elif np.random.random() > 0.75:
        image = augment_brightness_camera_images(image)
    elif np.random.random() > 0.75:
        image = augment_brightness_camera_images(image)
        image =  add_random_shadow(image)
        image, steer = horizontal_flip(image, steer)
                
    return image, steer

from itertools import chain
from itertools import islice

def batch_generator(file_names, batch_size):
    """
    Implement batch generator that yields items in batches of size batch_size.
    There's no need to shuffle input items, just chop them into batches.
    Remember about the last batch that can be smaller than batch_size!
    Input: any iterable (list, generator, ...). You should do `for item in items: ...`
        In case of generator you can pass through your items only once!
    Output: In output yield each batch as a list of items.
    """
    
    ### YOUR CODE HERE
    # https://stackoverflow.com/questions/24527006/split-a-generator-into-chunks-without-pre-walking-it
    iterator = iter(file_names)
    for first in iterator:        
        yield list(chain([first], islice(iterator, batch_size - 1)))


def normalize(image):
    '''Return image centered around 0 with +- 0.5.
    image: Image to transform in Numpy array.
    '''
    return image/255.-.5


def train_generator(file_names, batch_size, shuffle = True, jitter = True, norm=True):
  
    if shuffle: np.random.shuffle(file_names)
      
    while True:  # so that Keras can loop through this as long as it wants
        for batch in batch_generator(file_names, batch_size):
          
            # prepare batch images
            batch_imgs = []
            batch_targets = []
            for file_name in batch:
              
                image, steer = read_image_and_steering(file_name)
                
                if jitter: image, steer = augment(image, steer)
                if norm:   image = normalize(image)
                  
                batch_imgs.append(image)
                batch_targets.append(steer)
            # stack images into 4D tensor [batch_size, img_size, img_size, 3]
            batch_imgs = np.stack(batch_imgs, axis=0)
            # convert targets into 2D tensor [batch_size, num_classes]
            batch_targets = np.stack(batch_targets, axis=0)
            yield batch_imgs, batch_targets

IMG_SIZE = read_image_and_steering(files[0])[0].shape
IMG_SIZE


from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Model

def model_categorical(input_size= IMG_SIZE, dropout=0.1):
    '''Generate an NVIDIA AutoPilot architecture.
    Input_size: Image shape (90, 250, 3), adjust to your desired input.
    Dropout: Proportion of dropout used to avoid model overfitting.
    This model ONLY predicts steering angle as a 5-elements array encoded with a Softmax output.
    The model is already compiled and ready to be trained.
    '''

    
    img_in = Input(shape=input_size, name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = Convolution2D(nb_filter=24, nb_row=5, nb_col=5, subsample=(2,2), activation='relu')(img_in)       # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(nb_filter=32, nb_row=5, nb_col=5, subsample=(2,2), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(nb_filter=64, nb_row=5, nb_col=5, subsample=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)                                    # Classify the data into 100 features, make all negatives 0
    x = Dropout(dropout)(x)                                                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)                                     # Classify the data into 50 features, make all negatives 0
    x = Dropout(dropout)(x)                                                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    
    #categorical output of the angle
    angle_out = Dense(3, activation='softmax', name='angle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0
    
    model = Model(input=[img_in], output=[angle_out])
    
    return model
  
model = model_categorical()
model.summary()
    
from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop  = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=5, mode='auto', verbose=1)
checkpoint  = ModelCheckpoint('ironcar_weights_keras12.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)


from sklearn.model_selection import train_test_split

# batch generator
BATCH_SIZE = 32
#BATCH_SIZE = 16

files_train, files_test = train_test_split(files, test_size=0.15)

#num_train = len(imgs_num_train)//BATCH_SIZE
#num_valid = len(imgs_num_test)//BATCH_SIZE
num_train = len(files_train)
num_valid = len(files_test)


model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

model.fit_generator(generator = train_generator(files_train, BATCH_SIZE),
                    samples_per_epoch = num_train, 
                    nb_epoch  = 6, 
                    verbose = 1,
                    validation_data = train_generator(files_test, BATCH_SIZE, jitter = False), 
                    nb_val_samples = num_valid, 
                    callbacks = [early_stop, checkpoint])

