import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN


def img2vector(fileName):
    # 1 构建一个一维矩阵,用来存放数据
    returnVect = np.zeros((1, 1024))
    # 2 打开数据
    fr = open(fileName)
    # 3 将数据写入一维数组中
    #   3.1 按行读取数据
    for i in range (32):
        lineStr = fr.readline ()
        # 3.2 将每列的数据读取出来存放在数组中
        for j in range (32):
             returnVect[0, 32*i+j] = int(lineStr[j])
    #返回转换后的1x1024向量
    return returnVect

def handwritingClassTest():
    # 1 导入训练集与测试集数据
    trainingFileList = listdir ("trainingDigits")
    m = len (trainingFileList)
    # 2 建立一个 list 用来存放数据的标签
    hwLabels = []
    # 3 建立一个多少数组用来保存特征
    trainingMat = np.zeros ((m, 1024))
    # 4 从文件名中解析出训练集的类别
    for i in range(m):
        # 4.1 获得文件的名字
        fileNameStr = trainingFileList[i]
        # 4.2获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        # 4.3 将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        # 4.3将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i,:] = img2vector('trainingDigits/%s' % (fileNameStr))
    
    # 5 构建KNN分类器
    negih = KNN (n_neighbors=3, algorithm='auto')
    # 6 对数据进行拟合
    negih.fit (trainingMat, hwLabels)
    
    # 7 获取训练集数据
    testFileList = listdir ("testDigits")

    #错误检测计数
    errorCount = 0.0
    #测试数据的数量
    mTest = len(testFileList)
    #从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        #获得文件的名字
        fileNameStr = testFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #获得测试集的1x1024向量,用于训练
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        #获得预测结果
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = negih.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))
 
if __name__ == "__main__":
    handwritingClassTest()
    
