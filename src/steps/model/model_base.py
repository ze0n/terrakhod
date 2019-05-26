from os import listdir
from os.path import isfile, join
import cv2
import skimage.exposure as exposure
import random
from itertools import chain
from itertools import islice
import skimage
from skimage.io import imread, imshow
import numpy as np
import re
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


PATH = 'terrakhod-20181124/1/'

HYPERPARAMS = {
    "optimizer": 'adam',
    "loss": 'categorical_crossentropy',
    "metrics": ['accuracy']
}

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

  
model = model_categorical()
model.summary()
    
early_stop  = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=5, mode='auto', verbose=1)
checkpoint  = ModelCheckpoint('ironcar_weights_keras12.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)

# batch generator
BATCH_SIZE = 32
#BATCH_SIZE = 16

files_train, files_test = train_test_split(files, test_size=0.15)

#num_train = len(imgs_num_train)//BATCH_SIZE
#num_valid = len(imgs_num_test)//BATCH_SIZE
num_train = len(files_train)
num_valid = len(files_test)

model.compile(
    optimizer=HYPERPARAMS["optimizer"],
    loss=HYPERPARAMS["loss"],
    metrics=HYPERPARAMS['metrics'])

model.fit_generator(generator = train_generator(files_train, BATCH_SIZE),
                    samples_per_epoch = num_train, 
                    nb_epoch  = 6, 
                    verbose = 1,
                    validation_data = train_generator(files_test, BATCH_SIZE, jitter = False), 
                    nb_val_samples = num_valid, 
                    callbacks = [early_stop, checkpoint])

