import pandas as pd
import numpy as np
from PIL import Image
import io

def array_to_png(array):
    im = Image.fromarray(array)
    with io.BytesIO() as output:
        im.save(output, format="PNG")
        contents = output.getvalue()
    return contents

def array_to_image(array):
    im = Image.fromarray(array)
    return im
