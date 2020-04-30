from loader import *
import sys
import numpy as np
from PIL import Image, ImageDraw
import os
import numpy

class AxionautLoader(LoaderBase):
    def __init__(self, rootDir, outDir, datasetName, xfile, yfile):
        self.rootDir = rootDir
        self.outDir = outDir
        self.xfile = xfile
        self.yfile = yfile
        self.datasetName = datasetName
    
    def load(self):
        xfilePath = os.path.join(self.rootDir, self.xfile)
        yfilePath = os.path.join(self.rootDir, self.yfile)
        
        X = np.load(xfilePath)
        Xshape = X.shape

        Y = np.load(yfilePath)
        Yshape = Y.shape

        self.log(" "+"-"*80)
        self.log(" "+">> ", self.rootDir)
        self.log(" "+"-"*80)
        self.log(" X: {}".format(xfilePath))
        self.log(" Y: {}".format(yfilePath))
        self.log(" X shape:", X.shape)        
        self.log(" Y shape:", Y.shape)          
        self.log(" "+"-"*80)
        
        self.log("Cleaning X")
        del X
        self.log("Cleaning Y")
        del Y
        
        d = os.path.join(self.outDir, self.datasetName)
        self.log("Creating dataset dir {}".format(d))
        try:
            os.mkdir(d)
        except FileExistsError:
            pass
        
        from shutil import copyfile
        
        toXPath = os.path.join(d, "X.npy")
        if(os.path.isfile(toXPath)):
            self.log("X file exists - removing")
            os.unlink(toXPath)
        self.log("Copying X to {}".format(toXPath))
        copyfile(xfilePath, toXPath)

        toYPath = os.path.join(d, "Y.npy")
        if(os.path.isfile(toYPath)):
            self.log("Y file exists - removing")
            os.unlink(toYPath)
        self.log("Copying Y to {}".format(toYPath))
        copyfile(yfilePath, toYPath)

        with open(os.path.join(d, "dataset.txt"), "wt") as fout:
            fout.write("X: {}\n".format(Xshape))
            fout.write("Y: {}\n".format(Yshape))
        
        self.log("Done")

if(__name__ == "__main__"):
	L = AxionautLoader(*sys.argv[1:6])
	L.load()