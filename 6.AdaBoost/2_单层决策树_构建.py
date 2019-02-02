import numpy as np
import matplotlib.pyplot as plt

def loadSimpData ():
    dataMat = np.matrix ([[1., 2.1], 
                          [2., 1.1],
                          [1.3, 1.],
                          [1., 1.],
                          [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def showDataSet (dataMat, classLabels):
    data_plus  = []
    data_minus = []
    
    for i in range (len (classLabels)):
        if classLabels[i] > 0:
            data_plus.append (dataMat[i])
        else:
            data_minus.append (dataMat[i])
    data_plus_np = np.array(data_plus)                                             #转换为numpy矩阵
    data_minus_np = np.array(data_minus)                                         #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])        #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])     #负样本散点图
    plt.show()
    

def stumpClassify (dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones ((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'It':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal]  = 1.0
    return retArray

def buildStump (dataArr, classLabels, D):
    # 1 将数据向量化
    dataMatrix = np.mat (dataArr)
    labelMat   = np.mat (classLabels).T
    # 2 获取数据数和特征数
    m, n = np.shape (dataMatrix)
    # 3 定义步长
    numSteps = 10.0
    # 4 定义字典方便存储
    bestStump = {}
    bestClassEst = np.mat (np.zeros ((m, 1)))
    minError = float('inf')
    # 5 遍历所有特征
    for i in range (n):
        rangeMin = dataMatrix[:, i].min ()
        rangeMax = dataMatrix[:, i].max ()
        stepSize = (rangeMax - rangeMin) / numSteps
        # 5.1 寻找每个最佳决策点
        for j in range (-1, int(numSteps) - 1):
            # 少于 对于
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify (dataMatrix, i, threshVal, inequal)
                errArr = np.mat (np.ones ((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr                                      #计算误差
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:                                     #找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst
 
if __name__ == '__main__':
    dataArr,classLabels = loadSimpData()
    D = np.mat(np.ones((5, 1)) / 5)
    bestStump,minError,bestClasEst = buildStump(dataArr,classLabels,D)
    print('bestStump:\n', bestStump)
    print('minError:\n', minError)
    print('bestClasEst:\n', bestClasEst)
    
