import sys
import numpy as np
import pandas as pd

# sys.path("../../..")
# cwd = 

DATASETS_DIR = ""

class Labeler:

    def __init__(self, database, cursor_size):
        X = np.load('Datasets/axionable_data/X_train_axio.npy')
        Y = np.load('Datasets/axionable_data/Y_train_axio.npy')
        Y = np.load("predicted.npy")

        self.cursor_position = 0
        self.cursor_size = cursor_size
        self.cursor_mode = "continious"
        self.cursor = self.cursor_next(first=True)

    def cursor_next(self, first=False):
        index = self.X.index

        if(self.cursor_mode == "random"):
            self.cursor = index.shufle(self.cursor_size)

        elif(self.cursor_mode == "continious"):
            if(not first):
                self.current_position += self.cursor_size
            self.cursor = index[self.current_position:self.current_position + self.cursor_size]
        else:
            raise "Error"

    def cursor_render(self, current_position, window_size=1):
        im = show_image()

    def show_image(self, index):
        im = Image.fromarray(X[index])
        return im

    def render_labels(self, im, labels):
        piece_size = 20
        padding = 20
        text_padding = 30
        base_y = height - 10
        base_x = width / 2
        linewidth = 3

        for i in range(len(labels)):
            text, value = labels[i]

            canvas = ImageDraw.Draw(im)
            
            y_i = base_y + padding * i

            canvas.text(base_x - 2 * piece_size - text_padding, y_i, text)
            canvas.line((y[i] * piece_size + base_x, y_i), ((y[i] - 1.0) * piece_size + base_x, y_i ), fill=None, width=linewidth)
        
        return im
        #im.save(folder + "/" + str(index)+".jpeg")

    def load_labels(self, index, models=[]):
        ret = {}
        ret["orig"] = read_original_label(index)
        ret["corr"] = read_corrected_label(index)
        for model in models:
            ret[model] = render_predicted_label(model)
        return ret



    def show_image_with_labels(self):
        pass


    def read_original_label(self):
        pass

    def read_corrected_label(self):
        pass

    def read_predicted_label(self, model):
        pass

    def change_corrected_label(self, ):
        pass



print("Load data")
print("Predicted", Y[:10])

# (26449, 90, 250, 3)
shape = X_axio.shape
width = shape[2]
height = shape[1]

for index in range(shape[0])[:1000]:
    im = Image.fromarray(X_axio[index])
    y = Y[index]
    predict(im, y, index)

