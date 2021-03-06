import lrMethods as lrm
import mongoMat as mmat
import numpy as np
import matplotlib.pyplot as plt
import math
import pymongo
import configparser

def predict(x):
    mongourl = mmat.readProperties("application.properties", "mongo", "mongo-url")
    client = pymongo.MongoClient(mongourl)
    db = client.test
    coltheta = db["num-image-theta"]

    recent = coltheta.find().sort("timestamp", -1).limit(1)
    for row in recent:
        all_theta = np.asarray(row['thetas'])

    x = np.reshape(x, (1, x.size))
    p = lrm.predictToExpectation(all_theta, x)
    return p

def draw(x):
    side = math.floor(math.sqrt(x.size))
    arr = np.reshape(x, (side, side))
    plt.imshow(arr, cmap='gray', vmin=0, vmax=1)
    plt.show()
