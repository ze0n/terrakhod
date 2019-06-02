pipeline_parameters = {
    "checkpoint.filename": 'ironcar_weights_keras12.hdf5',
    "checkpoint.monitor": 'val_loss',
    "checkpoint.verbose": 1,
    "checkpoint.save_best_only": True,
    "checkpoint.mode": 'min',
    "checkpoint.period": 1,

    "IMG_SIZE": (90, 250, 3),
    "nb_epoch": 6,
    "BATCH_SIZE": 32,
    "optimizer": 'adam',
    "loss": 'categorical_crossentropy',
    "metrics": ['accuracy'],
    "datasets_path": "data/datasets"
}

model_parameters = {

}