# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:36:50 2019

@author: Hyzhang
"""

import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(path):
    """
    加载数据集
    X0设为1.0，构成拓充后的输入向量
    :return:输入向量矩阵和输出向量
    """
    dataMat = []; labelMat = []
    fr = open(path)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #X0设为1.0，构成拓充后的输入向量
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def plotBestFit(weights,dataMat,labelMat):
    """
    画出数据集和逻辑斯谛最佳回归直线
    :param weights:
    """    
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    if weights is not None:
        x = np.arange(-3.0, 3.0, 0.1)
        y = (-weights[0]-weights[1]*x)/weights[2]   #令w0*x0 + w1*x1 + w2*x2 = 0，其中x0=1，解出x1和x2的关系
        ax.plot(x, y)                               #一个作为X一个作为Y，画出直线
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def sigmoid(inX):
    """
    sigmoid函数
    """
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    """
    逻辑斯谛回归梯度上升优化算法
    :param dataMatIn:输入X矩阵（100*3的矩阵，每一行代表一个实例，每列分别是X0 X1 X2）
    :x0 为截距
    :param classLabels: 输出Y矩阵（类别标签组成的向量）
    :return:权值向量
    """
    dataMatrix = np.mat(dataMatIn)             #转换为 NumPy 矩阵数据类型
    labelMat = np.mat(classLabels).transpose() #转换为 NumPy 矩阵数据类型
    m,n = np.shape(dataMatrix)                 #矩阵大小
    alpha = 0.001                           #步长
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):              #最大迭代次数
        h = sigmoid(dataMatrix*weights)     #矩阵内积
        error = (labelMat - h)              #向量减法
        weights += alpha * dataMatrix.transpose() * error  #矩阵内积
    return weights

if __name__=='__main__':
    dataMat,labelMat=loadDataSet(unicode('E:/实战/李航统计学习方法/LR/testSet.txt','utf8'))
    plotBestFit(None,dataMat,labelMat)
    weights=gradAscent(dataMat,labelMat)
    plotBestFit(weights,dataMat,labelMat)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    