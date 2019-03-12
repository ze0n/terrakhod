import sys
import os
from loader import *
from loader_axionaut import *
from loader_shared import *


print("Downloading...")
if(not os.path.isfile("data/raw/downloaded.txt")):
	os.system("{}/download.cmd data/raw".format(os.path.dirname(os.path.realpath(__file__))))
	print("Downloading - Done")
print("Downloading - Skipped")

print("Loading...")

L = AxionautLoader("data/raw/2018-AXIONAUT/Datasets/axionable_data", "data/datasets", "axionaut-axionable_data", "X_train_axio.npy", "Y_train_axio.npy")
L.load()

L = AxionautLoader("data/raw/2018-AXIONAUT/Datasets/new", "data/datasets", "axionaut-new", "x_chicane.npy", "y_chicane.npy")
L.load()

L = AxionautLoader("data/raw/2018-AXIONAUT/Datasets/ironcar_data/new_track", "data/datasets", "axionaut-ironcar_data-new_track", "x_chicane.npy", "y_chicane.npy")
L.load()

L = AxionautLoader("data/raw/2018-AXIONAUT/Datasets/ironcar_data/old_track", "data/datasets", "axionaut-ironcar_data-old_track", "balanced_iron_X.npy", "balanced_iron_Y.npy")
L.load()

print("Loading - Done")

