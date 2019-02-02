# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 09:00:12 2018

@author: Administrator
"""

import re
import numpy as np
import random
"""
函数说明:接收一个大字符串并将其解析为字符串列表
 
Parameters:
    无
Returns:
    无
Author:
    NaLaEur
"""
def textParse(bigString):                                                   #将字符串转换为字符列表
    listOfTokens = re.split(r'\W*', bigString)                              #将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]            #除了单个字母，例如大写的I，其它单词变成小写
 
"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
 
Parameters:
    dataSet - 整理的样本数据集
Returns:
    vocabSet - 返回不重复的词条列表，也就是词汇表
Author:
    NaLaEur
"""
def createVocabList(dataSet):
    vocabSet = set([])                      #创建一个空的不重复列表
    for document in dataSet:               
        vocabSet = vocabSet | set(document) #取并集
    return list(vocabSet)

"""
3 函数说明:对每条言论进行one-hot编码
Parameters:
    vocabList - 词典
    inputSet  - 待处理言论
Returns:
    returnVec - one-hot编码
Author:
    NaLaEur
"""

# 3 将每句话中的数据转换为 one-hot 形式
def setOfWords2Vec(vocabList, inputSet):
    # 3.1 计算 vocabList 中一共有多少个单词
    nums = len (vocabList)
    # 3.2 创建一个一维矩阵将其转换为 (1, nums)的列表
    returnVec = [0] * nums
    # 3.3 遍历 inputSet 中的数据，将其转换为one-hot形式
    for word in inputSet:
        index = vocabList.index(word)
        returnVec[index] = 1
    return returnVec

"""
    4 函数说明:朴素贝叶斯分类器训练函数
     
    Parameters:
        trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
        trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
    Returns:
        p0Vect - 非侮辱类的条件概率数组  P (单词|非侮辱类)
        p1Vect - 侮辱类的条件概率数组    P (单词|侮辱类)
        pAbusive - 文档属于侮辱类的概率
    Author:
        NaLaEur
        
    改进 加入了拉普拉斯平滑
"""
def trainNB0(trainMatrix,trainCategory):
    # 1 计算出一共有多少样本数
    numTrainDocs = len (trainMatrix)
    # 2 计算出字典中一共有多少个数据，用来在下面计算出每个词在出现多少次
    numWords = len (trainMatrix[0])
    # 3 定义p0Num p1Num 
    p0Num   = np.ones (numWords)    # 不使用(1, numWords)，一个是一维数据，一个是多维数据
    p1Num   = np.ones (numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    
    # 4 开始进行循环，分别计算在侮辱类和非侮辱类词汇出现的次数
    for i in range (numTrainDocs):
        if (trainCategory[i] == 1):
            p1Num += trainMatrix[i]
            p1Denom += sum (trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum (trainMatrix[i])
    p1Vect = np.log (p1Num / p1Denom)
    p0Vect = np.log (p0Num / p0Denom)
    
    # 5 计算一共有多少侮辱言论的句子
    pAbusive = sum (trainCategory) / float (len (trainCategory))
    return p0Vect, p1Vect, pAbusive
    
"""
    5 函数说明 ： 对待测言论进行分类
    Parameters:
           vec2Classify - 待测样本的 one-hot 编码
           p0Vec        - P (单词 | 非侮辱类)
           p1Vec        - P (单词 | 侮辱类)
           pClass1      - P (侮辱类)
    return :
            侮辱类或者非侮辱类
            
        新增，使用log解决下溢出问题
"""
def classifyNB (vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)        #对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    
    print ("p1 : %d" % p1)
    print ("p0 : %d" % p0)
    if (p1 > p0):
        return 1
    else:
        return 0
    
def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):                                                  #遍历25个txt文件
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())     #读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)                                                 #标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())      #读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)                                                 #标记非垃圾邮件，1表示垃圾文件   
    vocabList = createVocabList(docList)                                    #创建词汇表，不重复
    trainingSet = list(range(50)); testSet = []                             #创建存储训练集的索引值的列表和测试集的索引值的列表                       
    for i in range(10):                                                     #从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))                #随机选取索索引值
        testSet.append(trainingSet[randIndex])                              #添加测试集的索引值
        del(trainingSet[randIndex])                                         #在训练集列表中删除添加到测试集的索引值
    trainMat = []; trainClasses = []                                        #创建训练集矩阵和训练集类别标签系向量             
    for docIndex in trainingSet:                                            #遍历训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))       #将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])                            #将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  #训练朴素贝叶斯模型
    errorCount = 0                                                          #错误分类计数
    for docIndex in testSet:                                                #遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])           #测试集的词集模型
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:    #如果分类错误
            errorCount += 1                                                 #错误计数加1
            print("分类错误的测试集：",docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))

if __name__ == "__main__":
    spamTest()