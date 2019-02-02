import numpy as np

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

# 2 创建一个链表
def createVocabList(dataSet):
    # 2.1 创建一个 set 链表用来存放元素
    returnVoc = set([])
    for doc in dataSet:
        temp_doc = set (doc)
        returnVoc = returnVoc | temp_doc
        
    return list (returnVoc)

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
    print (trainMat)
    print (myVocabList)
    