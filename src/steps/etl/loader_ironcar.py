from loader import *
import sys
import numpy as np
from PIL import Image, ImageDraw
import os
import numpy
import re

class IronCarLoader(LoaderBase):
    def __init__(self, rootDir, outDir, datasetName, thresholds):
        self.rootDir = rootDir
        self.outDir = outDir
        self.datasetName = datasetName
        
        V = []
        for i in range(5):
            V.append((thresholds[i], thresholds[i+1]))

        self.thresholds = V

    def directionMap(self, thresholds, direction):
        index = None
        for i in range(5):
            if(direction > thresholds[i][0] and direction < thresholds[i][1]):
                index = i
                break
        ret = [0] * 5
        ret[index] = 1
        return ret
    
    def load(self):
        files = os.listdir(self.rootDir)
        shape = None
        index = 0

        tuples = []

        for file in files:
            # frame_4537_gas_-0.4_dir_0.0.jpg
            frameNumber, gas, direction = re.match("frame_(\d+)_gas_([-+]?\d*\.\d+)_dir_([+-]?\d*\.\d+)\.jpg", file).groups()
            
            frameNumber = int(frameNumber)
            gas = float(gas)
            direction = float(direction)

            tuples.append((frameNumber, gas, direction, file))

        tuples = list(sorted(tuples))

        #print(tuples)

        Y = []
        X = []
        G = []
        D = []
        for t in tuples:

            frameNumber, gas, direction, file = t

            G.append(gas)
            D.append(direction)

            y_labels = self.directionMap(self.thresholds, direction)

            #print(file, y_labels)

            im = numpy.asarray(Image.open(os.path.join(self.rootDir, file)))

            #im = im[-110:-20,:,:]
            # (26449, 90, 250, 3)
            
            if(not shape == None):
                if(not shape == im.shape):
                    raise "Shape of images in the folder is not the same {}".format(file) 

            shape = im.shape

            X.append(im)
            Y.append(y_labels)
        
        Y = np.array(Y)
        X = np.array(X)
        
        G = np.array(G)
        D = np.array(D)

        Xshape = X.shape
        Yshape = Y.shape
        Gshape = G.shape
        Dshape = D.shape

        self.log(" "+"-"*80)
        self.log(" "+">> ", self.rootDir)
        self.log(" "+"-"*80)
        self.log(" X shape:", X.shape)        
        self.log(" Y shape:", Y.shape)          
        self.log(" G shape:", G.shape)        
        self.log(" D shape:", D.shape)          
        self.log(" "+"-"*80)
        
        d = os.path.join(self.outDir, self.datasetName)
        self.log("Creating dataset dir {}".format(d))
        try:
            os.mkdir(d)
        except FileExistsError:
            pass
        
        
        toXPath = os.path.join(d, "X.npy")
        if(os.path.isfile(toXPath)):
            self.log("X file exists - removing")
            os.unlink(toXPath)
        self.log("Saving X to {}".format(toXPath))
        np.save(toXPath, X)

        toYPath = os.path.join(d, "Y.npy")
        if(os.path.isfile(toYPath)):
            self.log("Y file exists - removing")
            os.unlink(toYPath)
        self.log("Saving Y to {}".format(toYPath))
        np.save(toYPath, Y)

        toGPath = os.path.join(d, "G.npy")
        if(os.path.isfile(toGPath)):
            self.log("G file exists - removing")
            os.unlink(toGPath)
        self.log("Saving G to {}".format(toGPath))
        np.save(toGPath, G)

        toDPath = os.path.join(d, "D.npy")
        if(os.path.isfile(toDPath)):
            self.log("D file exists - removing")
            os.unlink(toDPath)
        self.log("Saving D to {}".format(toDPath))
        np.save(toDPath, D)

        with open(os.path.join(d, "dataset.txt"), "wt") as fout:
            fout.write("X: {}\n".format(Xshape))
            fout.write("Y: {}\n".format(Yshape))
        
        self.log("Done")

if(__name__ == "__main__"):
	L = IronCarLoader(*sys.argv[1:5])
	L.load()