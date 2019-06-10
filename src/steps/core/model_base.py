from keras.callbacks import EarlyStopping, ModelCheckpoint

# "optimizer"
# "loss"
# "metrics"
from parameters import pipeline_parameters
from sklearn.model_selection import train_test_split
from src.steps.core.datasets import DatasetStorage, DatasetInfo

datasets_storage = DatasetStorage(pipeline_parameters["datasets_path"])
datasets_infos = datasets_storage.get_datasets_infos()
datasets = datasets_storage.get_datasets()

map(lambda x: x.image_size, datasets_infos)

# IMG_SIZE = read_image_and_steering(files[0])[0].shape
# IMG_SIZE
#
# model = model_categorical()
# model.summary()
#
# early_stop  = EarlyStopping(
#     monitor='val_loss',
#     min_delta=0.0005,
#     patience=5,
#     mode='auto',
#     verbose=1)
#
# checkpoint  = ModelCheckpoint(
#     pipeline_parameters["checkpoint.filename"],
#     monitor=pipeline_parameters['checkpoint.monitor'],
#     verbose=pipeline_parameters["checkpoint.verbose"],
#     save_best_only=pipeline_parameters["checkpoint.save_best_only"],
#     mode=pipeline_parameters["checkpoint.mode"],
#     period=pipeline_parameters["checkpoint.period"])
#
# # batch generator
# BATCH_SIZE = pipeline_parameters["BATCH_SIZE"]
#
# files_train, files_test = train_test_split(files, test_size=0.15)
#
# #num_train = len(imgs_num_train)//BATCH_SIZE
# #num_valid = len(imgs_num_test)//BATCH_SIZE
# num_train = len(files_train)
# num_valid = len(files_test)
#
# model.compile(
#     optimizer=pipeline_parameters["optimizer"],
#     loss=pipeline_parameters["loss"],
#     metrics=pipeline_parameters['metrics'])
#
# model.fit_generator(
#     generator = train_generator(files_train, BATCH_SIZE),
#     samples_per_epoch = num_train,
#     nb_epoch  = pipeline_parameters["nb_epoch"],
#     verbose = 1,
#     validation_data = train_generator(files_test, BATCH_SIZE, jitter = False),
#     nb_val_samples = num_valid,
#     callbacks = [early_stop, checkpoint])
#
