# num-recog-ml

An application using logistic regression to compute/recognize number in image(20X20).\
Use mat file as the initial samples base.\
Compute thetas and persist into mongodb.\
Read image(20X20) and compute prediction with thetas.

application.properties
```
[matlab]
matfile=XXX.mat

[mongo]
mongo-url=mongodb+srv://XXX:XXX@cluster.mongodb.net/XX?XXXX
```

[ex3data1Run.py](ex3data1Run.py) contains execution examples.
