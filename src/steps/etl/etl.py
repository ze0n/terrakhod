import sys
import os
from loader import *
from loader_axionaut import *
from loader_ironcar import *


print("Downloading...")
if(not os.path.isfile("data/raw/downloaded.txt")):
	os.system("{}/download.cmd data/raw".format(os.path.dirname(os.path.realpath(__file__))))
	print("Downloading - Done")
print("Downloading - Skipped")

print("Loading...")

axionautDatasestInfo = [
	("data/raw/2018-AXIONAUT/Datasets/axionable_data", "data/datasets", "axionaut-axionable_data", "X_train_axio.npy", "Y_train_axio.npy"),
	("data/raw/2018-AXIONAUT/Datasets/new", "data/datasets", "axionaut-new", "x_chicane.npy", "y_chicane.npy"),
	("data/raw/2018-AXIONAUT/Datasets/ironcar_data/new_track", "data/datasets", "axionaut-ironcar_data-new_track", "x_chicane.npy", "y_chicane.npy"),
	("data/raw/2018-AXIONAUT/Datasets/ironcar_data/old_track", "data/datasets", "axionaut-ironcar_data-old_track", "balanced_iron_X.npy", "balanced_iron_Y.npy")
]

for p in axionautDatasestInfo:
	L = AxionautLoader(*p)
	L.load()

ironcarDatasestInfo = [
	("data/raw/2018-IRONCAR-SHARED-1-250x150-JPEGS/good", "data/datasets", "ironcar-shared", 
		(-1000, -0.5, -0.1, +0.1 ,+0.5 ,+1000)),
	("data/raw/2018-IRONCAR-SHARED-2-250x150-JPEGS/car_repo/records", "data/datasets", "ironcar-friend-shared", 
		(-1000, -0.5, -0.1, +0.1 ,+0.5 ,+1000))
]

for p in ironcarDatasestInfo:
	L = IronCarLoader(*p)
	L.load()

print("Loading - Done")

