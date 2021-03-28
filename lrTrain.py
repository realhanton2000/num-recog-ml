import lrMethods as lrm
import mongoMat as mmat
import numpy as np
import pymongo
import configparser
import datetime

def train(X, y):
    num_labels = 10
    lambda_ = 0.1
    all_theta = lrm.lrTrain(X, y, 10, lambda_)
    return all_theta

def accuracy(all_theta, X, y):
    p = lrm.predictToExpectation(all_theta, X)
    p = np.reshape(p, (p.size,1))
    return np.average(p == y) * 100

def runjob():
    X, y = mmat.readFromMongo()
    all_theta = train(X, y)
    accuracy_ = accuracy(all_theta, X, y)

    mongourl = mmat.readProperties("application.properties", "mongo", "mongo-url")
    client = pymongo.MongoClient(mongourl)
    db = client.test
    coltheta = db["num-image-theta"]
    row = {"accuracy": accuracy_, "thetas": all_theta.tolist(), "timestamp": datetime.datetime.utcnow()}
    coltheta.insert_one(row)
