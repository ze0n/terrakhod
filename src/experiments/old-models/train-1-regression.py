
# coding: utf-8

# In[13]:


from os.path import isfile, join
import h5py
import pandas as pd
from os import listdir
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from skimage.color import rgb2gray
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python import keras
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, InputLayer, LeakyReLU, Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator # Usefull thing. Read the doc.
from tensorflow.python.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# In[14]:


IMG_SIZE = (90,180)
BATCH_SIZE = 32
EPOCHS = 100


# In[18]:


labels_dirname='../DatasetGeneration/SampleData/1'
img_dirname='../DatasetGeneration/SampleData/1'
csv_file = join(labels_dirname, 'data.csv')
df = pd.read_csv(csv_file, index_col=0)
df.head(5)


# In[19]:


df.Label.unique()


# In[20]:


labelMapping = {'Left5': -0.9, 'Left4': -0.8, 'Left3': -0.7, 'Left2': -0.6, 'Left1': -0.5, 'Straight': 0.0, 'Right1': 0.5, 'Right2': 0.6, 'Right3': 0.7, 'Right4': 0.8, 'Right5': 0.9}


# In[21]:


def load_imgs_and_labels(
    labels_dirname='../DatasetGeneration/SampleData/1', 
    img_dirname='../DatasetGeneration/SampleData/1', 
    debug=False):
  """
  Returns a numpy array of images, a numpy array of steering angles, and an array of labels
  """
  # Read csv
  csv_file = join(labels_dirname, 'data.csv')
  df = pd.read_csv(csv_file, index_col=0)
    
  # Read images
  img_files = [f for f in listdir(img_dirname) if f.endswith(".jpg") and isfile(join(img_dirname, f))]
    
  imgs = []
  labels = []
  for img_file in img_files:
    img = imread(join(img_dirname, img_file))
    # The original images are 1024x1280, way too big for the raspberry
    # I resize as in donkeycay: https://github.com/wroscoe/donkey/blob/dev/donkeycar/util/img.py
    # Obtained shape is (IMG_SIZE, IMG_SIZE, 3)
    
    # Take only lower part
    shape = img.shape
    img = img[int(shape[0]/2):]
    img_resized = resize(img, IMG_SIZE)
    # Convert to gray scale; obtained shape is (IMG_SIZE, IMG_SIZE)
    img_gray = rgb2gray(img_resized)
    # Need to add a dimension in order to get shape (IMG_SIZE, IMG_SIZE, 1), because the CNN
    # needs data with 3 or more dimensions
    img_reshaped = img_gray[..., np.newaxis]
    imgs.append(img_reshaped)
    # Get the corresponding label
    label = df.loc[img_file,'Label']
    labels.append(labelMapping[label])

    # I show the images for debugging purposes
    if ( debug ):
      imshow(img_gray)
      plt.title(img_file + ": " + label)
      plt.show()


  imgs_np = np.array(imgs)
                
  return imgs_np, labels


# In[26]:


def get_model(output_size):
  """
  Let's start with a simple model
  """

  model = keras.models.Sequential()
  # Define here your model

  model.add(Conv2D(filters=32, kernel_size=5, padding="same", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), activation='relu'))  # first layer needs to define "input_shape"
  #model.add(LeakyReLU(0.1))
  model.add(MaxPooling2D(pool_size = (2,2)))    
  model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation='relu'))
  #model.add(LeakyReLU(0.1))
  model.add(MaxPooling2D(pool_size = (2,2)))    
  model.add(Conv2D(filters=128, kernel_size=2, padding="same", activation='relu'))
  #model.add(LeakyReLU(0.1))
  model.add(MaxPooling2D(pool_size = (2,2)))
      
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  #model.add(LeakyReLU(0.1))
  model.add(Dropout(0.25))
  model.add(Dense(output_size, activation='sigmoid'))  

  return model


# In[27]:


def train_model(model, imgs, labels, model_name=None):

  imgs_train, imgs_val, labels_train, labels_val = train_test_split(imgs, labels, test_size=0.1)

  #model.compile(
  #    loss='categorical_crossentropy', 
  #    metrics=['accuracy'],
  #    optimizer=Adam()
  #)
  
  model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['mae', 'acc'])
  

  # Choose optimizer, compile model and run training
  earlyStopping = EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=2,
                                mode='auto')

  datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.1
                            )

  history = model.fit_generator(
      datagen.flow(imgs_train, labels_train, batch_size=BATCH_SIZE),
      validation_data=(imgs_val, labels_val),
      epochs=EPOCHS, 
      steps_per_epoch=len(imgs_train) // BATCH_SIZE,
      callbacks=[ModelCheckpoint(model_name + "-{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True),
                ] if model_name is not None else [],
      #callbacks=[earlyStopping],
      shuffle=True
      )  # starts training


# In[28]:


# Loads ims and labels
imgs, labels = load_imgs_and_labels(debug=False)
nb_different_labels = len(set(labels))


# In[12]:


labels


# In[29]:


model = get_model( 1 )
train_model(model, imgs, labels, model_name='Simple')

