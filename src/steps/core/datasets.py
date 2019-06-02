from abc import ABC, abstractmethod
from itertools import islice, chain
from typing import List

import numpy as np
import logging
import os

from src.steps.augumentation.augmentors import normalize


class DatasetStorage:
    def __init__(self, path, ds_filter:List[str]=None):
        self.ds_filter = ds_filter
        self.filter = filter
        self.path = path

    def list_datasets_names(self):
        ds_list = os.listdir(self.path)
        if(self.ds_filter != None):
            ds_list = filter(lambda x: x in self.ds_filter, ds_list)
        return ds_list

    def get_datasets(self):
        m = map(lambda x: Dataset(x, os.path.join(self.path, x)), self.list_datasets_names())
        return list(m)

    def get_datasets_infos(self):
        m = map(lambda x: Dataset(x, os.path.join(self.path, x)), self.list_datasets_names())
        infos = []
        for d in m:
            d.load()
            info = d.get_dataset_info()
            d.unload()
            infos.append(info)
        return infos

class DatasetInfo:
    def __init__(self, name, image_size, samples_count, path, corrected_samples_count):
        self.name = name
        self.corrected_samples_count = corrected_samples_count
        self.path = path
        self.samples_count = samples_count
        self.image_size = image_size

class SampleGenerator(ABC):
    @abstractmethod
    def batch_generator(self, batch_size):
        pass

class OneDatasetInMemorySampleGenerator(SampleGenerator):
    def __init__(self, dataset_storage : DatasetStorage):
        self.dataset_storage = dataset_storage
        self.dataset_infos = dataset_storage.get_datasets_infos()
        self.datasets = dataset_storage.get_datasets()
        self.indexed_datasets = {}
        for ds in self.datasets:
            self.indexed_datasets[ds.name] = ds
        self.cached_dataset : Dataset = None

    def batch_generator(self, multi_dataset_index, batch_size):
        iterator = iter(multi_dataset_index)
        for first in iterator:
            yield list(chain([first], islice(iterator, batch_size - 1)))


    def get_mlutidataset_index(self):

        multi_dataset_index = []

        for dataset_info in self.dataset_infos:
            dataset_index = zip([dataset_info.name] * dataset_info.samples_count,
                                range(dataset_info.samples_count))
            multi_dataset_index += dataset_index

        return multi_dataset_index

    def group_index_bydataset(self, multidataset_index):
        # [(a, 3), (a, 5), (b, 6)]
        indexed = {}
        for dsname, i in multidataset_index:
            if(not dsname in indexed):
                indexed[dsname] = []
            indexed[dsname].append(i)

        ret = []
        for dsname in sorted(indexed.keys()):
            for i in indexed[dsname]:
                ret.append((dsname, i))

        del indexed

        return ret


    def sample_generator(self, batch_size,
                         shuffle=True,
                         index_subset = None,
                         norm=True,
                         crop=None,
                         crop_strategy="bottom center"): # , jitter=True, norm=True):

        if(index_subset != None):
            multi_dataset_index = index_subset
        else:
            multi_dataset_index = self.get_mlutidataset_index()

        multi_dataset_index = self.group_index_bydataset(multi_dataset_index)

        if shuffle:
            raise Exception("Shufflung for multi-dataset is not implemented for now")

        while True:  # so that Keras can loop through this as long as it wants
            for batch in self.batch_generator(multi_dataset_index, batch_size):
                # prepare batch images
                batch_imgs = []
                batch_targets = []
                for dataset_name, dataset_index in batch:
                    image, steer = self.read_image_and_steering(dataset_name, dataset_index)

                    if crop != None:
                        height = crop[0]
                        width = crop[1]
                        cur_h = image.shape[0]
                        cur_w = image.shape[1]

                        if(crop_strategy == "bottom center"):
                            #print(image.shape)
                            assert(height <= cur_h and width <= cur_w)
                            h0 = cur_h - height
                            w0 = int((cur_w - width) / 2)
                            image = image[h0:,w0:,:]
                            #print(image.shape)

                            #raise Exception("PO")

                    #if jitter: image, steer = augment(image, steer)
                    if norm:   image = normalize(image)

                    batch_imgs.append(image)
                    batch_targets.append(steer)

                # stack images into 4D tensor [batch_size, img_size, img_size, 3]
                batch_imgs = np.stack(batch_imgs, axis=0)
                # convert targets into 2D tensor [batch_size, num_classes]
                batch_targets = np.stack(batch_targets, axis=0)

                yield batch_imgs, batch_targets

    def read_image_and_steering(self, dataset_name, dataset_index):
        dataset : Dataset = self.indexed_datasets[dataset_name]

        if(self.cached_dataset == None or self.cached_dataset.name != dataset_name):
            if(self.cached_dataset != None):
                self.cached_dataset.unload()
                logging.debug("Dataset {} was unloaded from the cache".format(self.cached_dataset.name))

            self.cached_dataset = dataset
            self.cached_dataset.load()
            logging.debug("Dataset {} was loaded to the cache".format(self.cached_dataset.name))

        x = dataset.X[dataset_index]

        # (1, 5)
        y = dataset.Y[dataset_index]
        c = dataset.C[dataset_index]

        if(not np.isnan(c).any()):
            y = c

        return x, y


class Dataset:
    def __init__(self, name, path):
        self.path = path
        self.name = name

    def get_dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            image_size=self.X[0].shape,
            samples_count=len(self.X),
            path=self.path,
            name=self.name,
            corrected_samples_count=sum(np.isnan(np.sum(self.C, axis=1)))
        )

    def load(self):
        self.X = np.load(os.path.join(self.path, 'X.npy'))
        logging.info(os.path.join(self.path, 'X.npy') + " loaded")
        self.Y = np.load(os.path.join(self.path, 'Y.npy'))
        logging.info(os.path.join(self.path, 'Y.npy') + " loaded")

        if(os.path.exists(os.path.join(self.path, 'C.npy'))):
            self.C = np.load(os.path.join(self.path, 'C.npy'))
            logging.info(os.path.join(self.path, 'C.npy') + " loaded")
        else:
            self.C = np.empty_like(self.Y)
            self.C[:] = np.nan
            logging.info(os.path.join(self.path, 'C.npy') + " created of nans")

    def save(self):
        np.save(os.path.join(self.path, 'X.npy'),self.X)
        logging.info(os.path.join(self.path, 'X.npy') + " saved")
        np.save(os.path.join(self.path, 'Y.npy'),self.Y)
        logging.info(os.path.join(self.path, 'Y.npy') + " saved")
        np.save(os.path.join(self.path, 'C.npy'),self.C)
        logging.info(os.path.join(self.path, 'C.npy') + " saved")

    def unload(self):
        del self.X
        del self.Y
        del self.C


