from PIL import Image
import computePredict
import os
import numpy as np

def readImage(filename):
    script_dir = os.path.dirname(__file__)
    abs_path = os.path.join(script_dir, filename)
    img = Image.open(abs_path).convert("L")
    img = np.array(img)
    img = transData(img)
    return img

def transData(x):
    x = x.flatten()
    x = x / 255
    return x
