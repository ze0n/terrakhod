import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict
from PIL import Image
import io
import base64

# sys.path("../../..")
# cwd = 

import logging

DATASETS_DIR = ""

class Dataset:
    def __init__(self, name, path):
        self.path = path
        self.name = name

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

class Labeler:

    def list_datasets(self, path:str)->List[Dict]:
        datasets = []

        for d in os.listdir(path):
            ds = {}
            ds["path"] = os.path.join(path, d)
            ds["name"] = d
            ds["description"] = open(os.path.join(path, d, "dataset.txt"), "rt").read()
            datasets.append(ds)

        return datasets

    def select_dataset(self, dataset:str) -> Dataset:
        
        self.dataset_description = self.datasets_indexed[dataset]

        if(self.dataset != None):
            self.dataset.save()
            self.dataset.unload()
            del self.dataset

        ret = Dataset(self.dataset_description["name"], self.dataset_description["path"])
        ret.load()

        self.dataset = ret

        self.current_position = 0
        self.cursor_history = []

    def __init__(self, datasets_path, cursor_size):

        print(os.getcwd())
        print(os.path.abspath(datasets_path))

        self.datasets_path = datasets_path
        self.datasets = self.list_datasets(self.datasets_path)

        self.datasets_indexed = {}
        for d in self.datasets:
            self.datasets_indexed[d["name"]] = d

        
        self.dataset = None

        if(len(self.datasets) == 0):
            raise Error("No datasets were found")

        self.select_dataset(self.datasets[0]["name"])
        
        self.current_position = 0
        self.cursor_size = cursor_size
        self.cursor_history = []
        self.cursor_mode = "continious"
        self.cursor = self.cursor_next(first=True)

    def cursor_next(self, first=False):
        index = np.arange(self.dataset.X.shape[0])

        if(self.cursor_mode == "random"):
            self.cursor = index.shufle(self.cursor_size)

        elif(self.cursor_mode == "continious"):
            if(not first):
                self.current_position += self.cursor_size
            self.cursor = index[self.current_position:self.current_position + self.cursor_size]
        else:
            raise "Error"

        self.cursor_history.append(self.current_position)


    def cursor_prev(self, first=False):
        pass

    def cursor_render(self):
        if(self.cursor_size == 1):
            im = self.show_image(self.current_position)
            labels = self.get_labels(self.current_position)
            return {
                "index": self.current_position,
                "image": im,
                "labels": labels
            }
        else:
            ret = list(map(lambda i: {
                "index": i,
                "image": self.show_image(i), 
                "labels": self.get_labels(i) 
                }, self.cursor))
            return ret

    def show_image(self, index):
        X = self.dataset.X
        im = Image.fromarray(X[index])

        with io.BytesIO() as output:
            im.save(output, format="PNG")
            contents = output.getvalue()

        return base64.b64encode(contents).decode("utf-8")

    def get_labels(self, index, models=[]):

        print(index)

        ret = {}
        ret["orig"] = self.dataset.Y[index]

        print(self.dataset.Y[index])

        if(np.isnan(self.dataset.C[index]).all()):
            ret["corr"] = np.empty_like(self.dataset.C[index])
            ret["corr"][:] = 0
        else:
            ret["corr"] = self.dataset.C[index]
        
        #for model in models:
        #    ret[model] = render_predicted_label(model)
        return ret

    def correct_label(self, dataset, imageIndex, newLabel):
        print("old: ", self.dataset.C[imageIndex])
        new = np.zeros(5)
        new[newLabel] = 1
        print("new: ", new)
        self.dataset.C[imageIndex] = new
        return "OK"

    def save(self):
        self.dataset.save()
        return "ok"

    # def render_labels(self, im, labels):
    #     piece_size = 20
    #     padding = 20
    #     text_padding = 30
    #     base_y = height - 10
    #     base_x = width / 2
    #     linewidth = 3

    #     for i in range(len(labels)):
    #         text, value = labels[i]

    #         canvas = ImageDraw.Draw(im)
            
    #         y_i = base_y + padding * i

    #         canvas.text(base_x - 2 * piece_size - text_padding, y_i, text)
    #         canvas.line((y[i] * piece_size + base_x, y_i), ((y[i] - 1.0) * piece_size + base_x, y_i ), fill=None, width=linewidth)
        
    #     return im
    #     #im.save(folder + "/" + str(index)+".jpeg")



# print("Load data")
# print("Predicted", Y[:10])

# # (26449, 90, 250, 3)
# shape = X_axio.shape
# width = shape[2]
# height = shape[1]

# for index in range(shape[0])[:1000]:
#     im = Image.fromarray(X_axio[index])
#     y = Y[index]
#     predict(im, y, index)

