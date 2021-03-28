import mongoMat
import lrTrain
import computePredict

def ex3data1():
    X, y = mongoMat.readFromMat()
    X = mongoMat.transData(X)
    mongoMat.writeInMongo(X, y)

def test():
    x, y = mongoMat.readFromMongo(1199)
    print(computePredict.predict(x))
    print(y)
    computePredict.draw(x)

#test()
#
#ex3data1()
#
#lrTrain.runjob()
