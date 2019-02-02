import numpy as np
from functools import reduce
# 1 创建数据
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]                                                                   #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec


"""
2 函数说明:根据现有言论构建一个词典
Parameters:
    dataSet - 训练文档矩阵   
Returns:
    returnVoc - 词典
Author:
    NaLaEur
"""

# 2 创建一个词典
def createVocabList(dataSet):
    # 2.1 创建一个 set 链表用来存放元素
    returnVoc = set([])
    for doc in dataSet:
        temp_doc = set (doc)
        returnVoc = returnVoc | temp_doc
        
    return list (returnVoc)

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
"""
def trainNB0(trainMatrix,trainCategory):
    # 1 计算出一共有多少样本数
    numTrainDocs = len (trainMatrix)
    # 2 计算出字典中一共有多少个数据，用来在下面计算出每个词在出现多少次
    numWords = len (trainMatrix[0])
    # 3 定义p0Num p1Num 
    p0Num   = np.zeros (numWords)    # 不使用(1, numWords)，一个是一维数据，一个是多维数据
    p1Num   = np.zeros (numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    
    # 4 开始进行循环，分别计算在侮辱类和非侮辱类词汇出现的次数
    for i in range (numTrainDocs):
        if (trainCategory[i] == 1):
            p1Num += trainMatrix[i]
            p1Denom += sum (trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum (trainMatrix[i])
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    
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
"""
def classifyNB (vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = reduce (lambda x, y : x * y, vec2Classify * p1Vec) * pClass1
    p0 = reduce (lambda x, y : x * y, vec2Classify * p0Vec) * (1 - pClass1)
    
    print ("p1 : %d" % p1)
    print ("p0 : %d" % p0)
    if (p1 > p0):
        return 1
    else:
        return 0
    
if __name__ == '__main__':
    
    # 1 创建数据
    postingLIst, classVec = loadDataSet()
    # 2 创建一个词典
    myVocabList = createVocabList (postingLIst)
    # 3 创建一个矩阵，里面存放着每句话在词典中的one-hot形式
    trainMat = []
    for doc in postingLIst:
        returnVec = setOfWords2Vec (myVocabList, doc)
        trainMat.append (returnVec)
    
    p0V, p1V, pAb = trainNB0 (trainMat, classVec)
    """
    print('p0V:\n', p0V)
    print('p1V:\n', p1V)
    print('classVec:\n', classVec)
    print('pAb:\n', pAb)
    """
    testEntry = ['love', 'my', 'dalmation']
    
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')		

    """
        这里可以发现会出现错误，P1 = P0 = 0，因为采用的是累积运算，有一个0会全盘0
    """
    testEntry = ['dog', 'stupid']
    
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')										
