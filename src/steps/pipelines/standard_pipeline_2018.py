import numpy as np
import sys

from keras.callbacks import ModelCheckpoint, EarlyStopping

sys.path.append(".")
import pandas as pd

import logging

from src.steps.augumentation.augmentors import adjust_gamma_dark, night_effect, add_random_shadow, \
    augment_brightness_camera_images, horizontal_flip
from src.steps.models.nvidia_autopilot import model_categorical

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.ERROR)

def Augment(image, steer):
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
        image = add_random_shadow(image)
        image, steer = horizontal_flip(image, steer)

    return image, steer



#from keras.callbacks import EarlyStopping, ModelCheckpoint

# "optimizer"
# "loss"
# "metrics"
from src.steps.core.parameters import pipeline_parameters
from sklearn.model_selection import train_test_split
from src.steps.core.datasets import DatasetStorage, DatasetInfo, OneDatasetInMemorySampleGenerator

###################################
# LOAD DATASETS INFORMATION
###################################

# 0          axionaut-axionable_data          data/datasets\axionaut-axionable_data   (90, 250, 3)          26449                    26434
# 1  axionaut-ironcar_data-new_track  data/datasets\axionaut-ironcar_data-new_track   (90, 250, 3)           1519                        0
# 2  axionaut-ironcar_data-old_track  data/datasets\axionaut-ironcar_data-old_track   (90, 250, 3)          16028                        0
# 3                     axionaut-new                     data/datasets\axionaut-new   (90, 250, 3)           3169                        0
# 4            ironcar-friend-shared            data/datasets\ironcar-friend-shared  (150, 250, 3)           4074                        0
# 5                   ironcar-shared                   data/datasets\ironcar-shared  (150, 250, 3)           4545                        0

datasets_storage = DatasetStorage(pipeline_parameters["datasets_path"], ds_filter=["axionaut-axionable_data", "ironcar-shared"])
datasets_infos = datasets_storage.get_datasets_infos()
datasets = datasets_storage.get_datasets()

print("==================\n Datasets\n==================\n")
print(pd.DataFrame(map(lambda x: x.__dict__, datasets_infos))[["name", "path", "image_size", "samples_count", "corrected_samples_count"]])

# Check dataset dimensions
df_dims = pd.DataFrame(map(lambda x: x.image_size, datasets_infos))
df_dims.columns = ["H", "W", "B"]
crop_window = []
for dim in df_dims.columns:
    widths = df_dims[dim].unique()
    if len(widths) > 1:
        logging.info("Datasets have different {} dimensions will automatically crop to minimal:".format(dim) + str(np.min(widths)))
    crop_window.append(np.min(widths))

crop_window = tuple(crop_window)
logging.info("Crop window is {}".format(crop_window))

###################################
# SPLIT TRAIN / TEST
###################################
sample_generator = OneDatasetInMemorySampleGenerator(dataset_storage=datasets_storage)
large_index = sample_generator.get_mlutidataset_index()

index_train, index_test = train_test_split(large_index, test_size=0.15)

num_train = len(index_train)
num_valid = len(index_test)

###################################
# BUILD A LAZY GENERATOR
###################################
train_generator = OneDatasetInMemorySampleGenerator(dataset_storage=datasets_storage)
test_generator = OneDatasetInMemorySampleGenerator(dataset_storage=datasets_storage)

###################################
# BUILD A MODEL
###################################
model = model_categorical(input_size=crop_window)
model.summary()

model.compile(
    optimizer=pipeline_parameters["optimizer"],
    loss=pipeline_parameters["loss"],
    metrics=pipeline_parameters['metrics'])

###################################
# TRAINING PARAMS
###################################
early_stop  = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0005,
    patience=5,
    mode='auto',
    verbose=1)

checkpoint = ModelCheckpoint(
    pipeline_parameters["checkpoint.filename"],
    monitor=pipeline_parameters['checkpoint.monitor'],
    verbose=pipeline_parameters["checkpoint.verbose"],
    save_best_only=pipeline_parameters["checkpoint.save_best_only"],
    mode=pipeline_parameters["checkpoint.mode"],
    period=pipeline_parameters["checkpoint.period"])

# batch generator
BATCH_SIZE = pipeline_parameters["BATCH_SIZE"]

###################################
# FIT
###################################
model.fit_generator(
    generator = train_generator.sample_generator(BATCH_SIZE, shuffle=False, index_subset=index_train, crop=crop_window),
    samples_per_epoch = num_train,
    nb_epoch  = pipeline_parameters["nb_epoch"],
    verbose = 1,
    validation_data = test_generator.sample_generator(BATCH_SIZE, shuffle=False, index_subset=index_test, crop=crop_window),
    nb_val_samples = num_valid,
    callbacks = [early_stop, checkpoint])

