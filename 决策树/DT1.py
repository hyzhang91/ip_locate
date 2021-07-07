# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:52:26 2018

@author: Hyzhang
"""

import matplotlib.pyplot as plt
import pandas as pd
import operator
from math import *

def majorityCnt(classList):
    """
返回出现次数最多的分类名称
    :param classList: 类列表
    :return: 出现次数最多的类名称
    """
    classCount = {}  # 这是一个字典
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
 
 


def createDataSet():
    """
创建数据集
 
    :return:
    """
    dataSet = [[u'青年', u'否', u'否', u'一般', u'拒绝'],
               [u'青年', u'否', u'否', u'好', u'拒绝'],
               [u'青年', u'是', u'否', u'好', u'同意'],
               [u'青年', u'是', u'是', u'一般', u'同意'],
               [u'青年', u'否', u'否', u'一般', u'拒绝'],
               [u'中年', u'否', u'否', u'一般', u'拒绝'],
               [u'中年', u'否', u'否', u'好', u'拒绝'],
               [u'中年', u'是', u'是', u'好', u'同意'],
               [u'中年', u'否', u'是', u'非常好', u'同意'],
               [u'中年', u'否', u'是', u'非常好', u'同意'],
               [u'老年', u'否', u'是', u'非常好', u'同意'],
               [u'老年', u'否', u'是', u'好', u'同意'],
               [u'老年', u'是', u'否', u'好', u'同意'],
               [u'老年', u'是', u'否', u'非常好', u'同意'],
               [u'老年', u'否', u'否', u'一般', u'拒绝'],
               ]
    labels = [u'年龄', u'有工作', u'有房子', u'信贷情况']
    # 返回数据集和每个维度的名称
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    """
按照给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征的维度
    :param value: 特征的值
    :return: 符合该特征的所有实例（并且自动移除掉这维特征）
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 删掉这一维特征
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def calcShannonEnt(dataSet):
    """
计算训练数据集中的Y随机变量的香农熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)  # 实例的个数
    labelCounts = {}
    for featVec in dataSet:  # 遍历每个实例，统计标签的频次
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt

def calcConditionalEntropy(dataSet, i, featList, uniqueVals):
    '''
    计算X_i给定的条件下，Y的条件熵
    :param dataSet:数据集
    :param i:维度i
    :param featList: 数据集特征列表
    :param uniqueVals: 数据集特征集合
    :return:条件熵
    '''
    ce = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)
        prob = len(subDataSet) / float(len(dataSet))  # 极大似然估计概率
        ce += prob * calcShannonEnt(subDataSet)  # ∑pH(Y|X=xi) 条件熵的计算
    return ce

def calcInformationGain(dataSet, baseEntropy, i):
    """
    计算信息增益
    :param dataSet:数据集
    :param baseEntropy:数据集中Y的信息熵
    :param i: 特征维度i
    :return: 特征i对数据集的信息增益g(dataSet|X_i)
    """
    featList = [example[i] for example in dataSet]  # 第i维特征列表
    uniqueVals = set(featList)  # 转换成集合
    newEntropy = calcConditionalEntropy(dataSet, i, featList, uniqueVals)
    infoGain = baseEntropy - newEntropy  # 信息增益，就是熵的减少，也就是不确定性的减少
    return infoGain

def calcInformationGainRate(dataSet, baseEntropy, i):
    """
    计算信息增益比
    :param dataSet:数据集
    :param baseEntropy:数据集中Y的信息熵
    :param i: 特征维度i
    :return: 特征i对数据集的信息增益g(dataSet|X_i)
    """
    return calcInformationGain(dataSet, baseEntropy, i) / baseEntropy

def chooseBestFeatureToSplitByID3(dataSet):
    """
    选择最好的数据集划分方式
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1  # 最后一列是分类
    baseEntropy = calcShannonEnt(dataSet) #计算类的熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有维度特征
        infoGain = calcInformationGain(dataSet, baseEntropy, i)
        if (infoGain > bestInfoGain):  # 选择最大的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回最佳特征对应的维度
    

def createTree(dataSet, labels, chooseBestFeatureToSplitFunc=chooseBestFeatureToSplitByID3):
    """
创建决策树
    :param dataSet:数据集
    :param labels:数据集每一维的名称
    :return:决策树
    """
    classList = [example[-1] for example in dataSet]  # 类别列表
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 当类别完全相同则停止继续划分
    if len(dataSet[0]) == 1:  # 当只有一个特征的时候，遍历完所有实例返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplitFunc(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 复制操作
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

    
#绘制带箭头的注释
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
 
 
#计算叶结点数
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs
 
 
#计算树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth
 
 
#在父子结点间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
 
 
def plotTree(myTree, parentPt, nodeTxt,totalW,totalD,xOff,yOff):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (xOff + (1.0 + float(numLeafs)) / 2.0 / totalW, yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)  #在父子结点间填充文本信息
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  #绘制带箭头的注释
    secondDict = myTree[firstStr]
    yOff = yOff - 1.0 / totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            xOff = xOff + 1.0 / totalW
            plotNode(secondDict[key], (xOff, yOff), cntrPt, leafNode)
            plotMidText((xOff, yOff), cntrPt, str(key))
    yOff = yOff + 1.0 / totalD
    
#def createPlot(inTree):
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    axprops = dict(xticks=[], yticks=[])
#    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
#    totalW = float(getNumLeafs(inTree))
#    totalD = float(getTreeDepth(inTree))
#    xOff = -0.5 / totalW
#    yOff = 1.0
#    plotTree(inTree, (0.5, 1.0), '')
#    plt.show()

##################################
 
# 测试决策树的构建
if __name__=='__main__':
    # global totalW
    # global totalD
    # global xOff
    # global yOff
    myDat, labels = createDataSet()
    myTree = createTree(myDat, labels)
    # 绘制决策树
    decisionNode = dict(boxstyle="round4", color='#3366FF')  #定义判断结点形态
    leafNode = dict(boxstyle="circle", color='#FF6633')  #定义叶结点形态
    arrow_args = dict(arrowstyle="<-", color='g')  #定义箭头
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    ax1 = plt.subplot(111, frameon=False, **axprops)
    totalW = float(getNumLeafs(myTree))
    totalD = float(getTreeDepth(myTree))
    xOff = -0.5 / totalW
    yOff = 1.0
    plotTree(myTree, (0.5, 1.0), '',totalW,totalD,xOff,yOff)
    plt.show()
    
    


##
#with open(unicode('E:/实战/李航统计学习方法/决策树/example_data.txt','utf8'),'w') as f:
#    for line in myDat:
#        for i in range(len(line)):
#            if i!=len(line)-1:
#                f.write(line[i].encode('utf8'))
#                f.write('\t')
#            else:
#                f.write(line[i].encode('utf8'))
#                f.write('\n')
#
#df=pd.read_csv(unicode('E:/实战/李航统计学习方法/决策树/example_data.txt','utf8'),sep='\t')






