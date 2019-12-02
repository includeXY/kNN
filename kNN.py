from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX, dataSet, labels, k):
    """
    :param inX: 未知类别属性点 ，是array类型
    :param dataSet: 训练数据集中的属性点，是array类型
    :param labels: 训练数据集中的标签，是list类型
    :param k: 选择与inX距离最小的前k个点
    :return: 返回前k个点中同一类别最多的那一类标签
    """
    dataSetSize = dataSet.shape[0]       #array_name.shape返回数组的行列(m,n)元组
    diffMat = tile(inX,(dataSetSize,1)) - dataSet       #得到未知类别点与所有已知类别点的坐标差
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)                #axis=1表示按行求和，=0表示按列求和
    distances = sqDistances**0.5
    sortedDisIndicies = distances.argsort()            #返回递增排序的索引，类型为array
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]      #选择与inX距离最小的k个点的类别
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  #查找字典classCount中是否存在关键值voteIlabel，没有返回0
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #key=operator.itemgetter(1)表示根据第二个域（类别数量）逆序排序,classCount.items()返回字典的键值对列表
    return sortedClassCount[0][0]