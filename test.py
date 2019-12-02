import kNN
from numpy import *
group,labels = kNN.createDataSet()
print(group)
print(labels)

label = kNN.classify0([0,0], group, labels, 3)
print(label)
