import pymongo
import scipy.io as sio
import os
import numpy as np
import configparser

def readProperties(filename, key, subkey):
    config = configparser.RawConfigParser()
    script_dir = os.path.dirname(__file__)
    abs_path = os.path.join(script_dir, filename)
    config.read(abs_path)
    return config[key].get(subkey)

def writeInMongo():
    script_dir = os.path.dirname(__file__)
    rel_path = readProperties("application.properties", "matlab", "matfile")
    abs_path = os.path.join(script_dir, rel_path)

    mat_contents = sio.loadmat(abs_path)
    X = mat_contents['X']
    y = mat_contents['y']
    X = abs(X)

    mongourl = readProperties("application.properties", "mongo", "mongo-url")
    client = pymongo.MongoClient(mongourl)
    db = client.test
    col = db["num-image-xy"]

    numrow, numcol = X.shape
    
    for index in range(numrow):
        X_line = X[index];
        y_line = y[index];
        row = {"y": y_line.tolist(), "x": X_line.tolist()}
        col.insert_one(row)
        print(index)

def readFromMongo():
    mongourl = readProperties("application.properties", "mongo", "mongo-url")
    client = pymongo.MongoClient(mongourl)
    db = client.test
    col = db["num-image-xy"]

    cursor = col.find({})
    X_new = []
    y_new = []
    index = 0
    for row in cursor:
        X_new.append(np.fromiter(row['x'], float))
        y_new.append(np.fromiter(row['y'], int))
        index += 1
        print(index)
    
    return np.asarray(X_new), np.asarray(y_new)


writeInMongo()

X, y = readFromMongo()