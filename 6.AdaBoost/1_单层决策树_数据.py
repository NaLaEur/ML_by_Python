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
    
if __name__ == "__main__":
    dataMat, classLaels = loadSimpData()
    showDataSet (dataMat, classLaels)
    