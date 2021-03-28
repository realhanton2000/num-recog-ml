import pymongo
import scipy.io as sio
import os
import numpy as np
import configparser
import datetime
import math
import logging
logging.basicConfig(level=logging.WARN)

def readProperties(filename, key, subkey):
    config = configparser.RawConfigParser()
    script_dir = os.path.dirname(__file__)
    abs_path = os.path.join(script_dir, filename)
    config.read(abs_path)
    return config[key].get(subkey)

def readFromMat():
    script_dir = os.path.dirname(__file__)
    rel_path = readProperties("application.properties", "matlab", "matfile")
    abs_path = os.path.join(script_dir, rel_path)

    mat_contents = sio.loadmat(abs_path)
    X = mat_contents['X']
    y = mat_contents['y']
    return X, y

def transData(X):
    numrow, numcol = X.shape
    side = math.floor(math.sqrt(numcol))

    X_new = np.empty([numrow, numcol])

    for index in range(numrow):
        X_line = X[index]
        #scale X into -1:1
        X_line_max = np.amax(abs(X_line))
        X_line = X_line / X_line_max
        #scale X into 0:1
        X_line = X_line / 2 + 0.5
        #transpose matrix
        X_mat = np.reshape(X_line, (side, side))
        X_mat = X_mat.transpose()
        #flat into ndarray
        X_line = X_mat.flatten()
        X_new[index] = X_line
    return X_new

def writeInMongo(X, y):
    mongourl = readProperties("application.properties", "mongo", "mongo-url")
    client = pymongo.MongoClient(mongourl)
    db = client.test
    col = db["num-image-xy"]

    numrow, numcol = X.shape
    
    for index in range(numrow):
        X_line = X[index]
        y_line = y[index]
        row = {"y": y_line.tolist(), "x": X_line.tolist(), "row": index, "timestamp": datetime.datetime.utcnow()}
        col.insert_one(row)
        logging.debug("writeInMongo - " + str(index))

def readFromMongo(index=-1):
    mongourl = readProperties("application.properties", "mongo", "mongo-url")
    client = pymongo.MongoClient(mongourl)
    db = client.test
    col = db["num-image-xy"]

    if (index < 0):
        cursor = col.find({})
        X_new = []
        y_new = []
        index = 0
        for row in cursor:
            X_new.append(np.fromiter(row['x'], float))
            y_new.append(np.fromiter(row['y'], int))
            index += 1
            logging.debug("readFromMongo - " + str(index))
        return np.asarray(X_new), np.asarray(y_new)
    else:
        cursor = col.find({ "row": index })
        X_new = []
        y_new = []
        for row in cursor:
            X_new.append(np.fromiter(row['x'], float))
            y_new.append(np.fromiter(row['y'], int))
        return np.asarray(X_new), np.asarray(y_new)

def ex3data1():
    X, y = readFromMat()
    X = transData(X)
    writeInMongo(X, y)
