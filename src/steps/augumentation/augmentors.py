from typing import Any, Dict, List, Tuple

import numpy as np
import re
from os import listdir
from os.path import isfile, join
import skimage
from skimage.io import imread, imshow
import skimage.exposure as exposure
import random
import cv2
from itertools import chain
from itertools import islice
import sys

sys.path.append(".")

AUG_PARAM_BLUR_KRNL_SIZE = "blur_kernel_size"
DEBUG = "debug"


# ----------------------------
# augmentor(image, steer, params) -> List[(image, steer)]
# ----------------------------

def read_param(param: str, params: Dict, default: Any):
    value = default
    if (param in params):
        value = params[param]
    return value


from IPython.display import Image, display, HTML
from src.steps.core.visualize import array_to_image

class AugmentationPipeline():
    _augmentors = []

    def __init__(self, augmentors:List):
        self._augmentors = zip(augmentors,[False] * len(augmentors))

    def add_step(self, augmentor, spawn:bool=False):
        self._augmentors.append((augmentor, spawn))

    def run_for_one_image(self, image: np.ndarray, steer: np.ndarray, params={}, debug: bool = False, no_spawn:bool=False):

        all_results = []

        for augmentor, spawn in self._augmentors:

            results = augmentor(image=image, steer=steer, params=params)

            all_results += results

            if(spawn and (not no_spawn)):


            if (debug):
                for new_image, new_steer in results:
                    display(HTML("<div>%s &rarr; %s</div>" % (augmentor.__name__, new_steer)))
                    display(array_to_image(new_image))

    def run(self, images: np.ndarray, steers: np.ndarray, params={}, debug: bool = False):
        X = images
        Y = steers
        for index in range(len(X)):
            self.run_for_one_image(image=X[index], steer=Y[index], params=params, debug=debug)


def horizontal_flip(image: np.ndarray, steer: np.ndarray, params={}) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Horizontal image flipping and angle correction.
    Img: Input image to transform in Numpy array.
    Angle: Corresponding label. Must be a 5-elements Numpy Array.
    """
    return [(np.fliplr(image), np.flipud(steer))]


def blur(image: np.ndarray, steer: np.ndarray, params={}) -> List[Tuple[np.ndarray, np.ndarray]]:
    kernel_size = read_param(AUG_PARAM_BLUR_KRNL_SIZE, params, 5)
    return [(cv2.blur(image, (kernel_size, kernel_size)), steer)]


def augment_brightness_camera_images(image: np.ndarray, steer: np.ndarray, params={}) -> List[
    Tuple[np.ndarray, np.ndarray]]:
    '''Random bright augmentation (both darker and brighter).
    Returns:
    Transformed image and label.
    '''
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return [(image1, steer)]


def add_random_shadow(image: np.ndarray, steer: np.ndarray, params={}) -> List[Tuple[np.ndarray, np.ndarray]]:
    '''Add random dark shadows to a given image.
    Returns:
    Transformed image and label.
    '''
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    # random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return [(image, steer)]


NIGHT_EFFECT_VMIN = "NIGHT_EFFECT_VMIN"
NIGHT_EFFECT_VMAX = "NIGHT_EFFECT_VMAX"


def night_effect(image: np.ndarray, steer: np.ndarray, params={}) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Change road color to black simulating night road.
    Returns
    Transformed image and label.
    """
    vmin = read_param(NIGHT_EFFECT_VMIN, params, 185)
    vmax = read_param(NIGHT_EFFECT_VMAX, params, 195)
    limit = random.uniform(vmin, vmax)
    low_limit = 146
    int_img = exposure.rescale_intensity(image, in_range=(low_limit, limit), out_range='dtype')

    return [(int_img, steer)]


def adjust_gamma_dark(image, steer, min_=0.7, max_=0.8):
    '''Gamma correction to generate darker images.
    Image: Image in Numpy format (90,250,3)
    Label: Corresponding label of the image.
    Min: Minimum gamma value (the lower the darker)
    Max: Maximum gamma value (the higher the brigther) 
    Return:
    Transformed image and label.
    '''
    # build a lookup table mapping the pixel values [0, 255] to
    gamma = random.uniform(min_, max_)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table), steer


def batch_generator(file_names, batch_size):
    """
    Implement batch generator that yields items in batches of size batch_size.
    There's no need to shuffle input items, just chop them into batches.
    Remember about the last batch that can be smaller than batch_size!
    Input: any iterable (list, generator, ...). You should do `for item in items: ...`
        In case of generator you can pass through your items only once!
    Output: In output yield each batch as a list of items.
    """

    ### YOUR CODE HERE
    # https://stackoverflow.com/questions/24527006/split-a-generator-into-chunks-without-pre-walking-it
    iterator = iter(file_names)
    for first in iterator:
        yield list(chain([first], islice(iterator, batch_size - 1)))


def normalize(image):
    '''Return image centered around 0 with +- 0.5.
    image: Image to transform in Numpy array.
    '''
    return image / 255. - .5
