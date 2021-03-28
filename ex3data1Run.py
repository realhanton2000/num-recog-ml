import mongoMat
import lrTrain
import computePredict
import imagePix

def ex3data1():
    X, y = mongoMat.readFromMat()
    X = mongoMat.transData(X)
    mongoMat.writeInMongo(X, y)

def test():
    x, y = mongoMat.readFromMongo(1199)
    print(computePredict.predict(x))
    print(y)
    computePredict.draw(x)

def test2():
    x = imagePix.readImage("test.bmp")
    print(computePredict.predict(x))
    computePredict.draw(x)

### test fucntion
#test()
#test2()

### init mongo db with data from mat
#ex3data1()

### compute and persist thetas
#lrTrain.runjob()


