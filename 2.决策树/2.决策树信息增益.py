from math import log

# 1 构建数据集
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],         #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['不放贷', '放贷']             #分类属性
    return dataSet, labels                #返回数据集和分类属性
# 2 计算熵值
def calcShannonEnt (dataSet):
    # 2.1 知晓数据集数量
    numEntries = len (dataSet)
    # 2.2 声明一个字典，用来存放每种类型数据的个数
    labelCounts = {}
    # 2.3 开始计算每种类型数据的个数
    for featVec in dataSet:
        currentLable = featVec[-1]
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable] = 0
        labelCounts[currentLable] += 1
    # 2.4 开始计算熵值
    shannonEnt = 0.0
    for key in labelCounts:
        probs = float (labelCounts[key]) / numEntries
        shannonEnt -= probs * log (probs, 2)
        
    return shannonEnt
def splitDataSet(dataSet, axis, value):       
    retDataSet = []                                        #创建返回的数据集列表
    for featVec in dataSet:                             #遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]                #去掉axis特征
            reducedFeatVec.extend(featVec[axis+1:])     #将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet                                      #返回划分后的数据集

# 3 计算最好的特征增益
def chooseBestFeatureToSplit (dataSet):
    # 3.1 计算特征值数量
    numFeatures = len (dataSet[0]) - 1
    # 3.2 获取基准信息熵
    baseEntropy = calcShannonEnt (dataSet)
    # 最大的信息增益
    bestInfoGain = 0.0
    bestFeature  = -1
    # 3.3 计算每个特征下的信息增益
    for i in range (numFeatures):
        # 3.3.1 提取出每个特征包含的特征，用于下面计算特征增益时候的概率
        featList = [example[i] for example in dataSet]
        uniqueVals = set (featList)
        newEntropy = 0.0
        for val in uniqueVals:
            # 将数据进行切分,关于每个特征的每个取值
            subDataSet = splitDataSet (dataSet, i, val)
            # 计算前缀
            prob = len (subDataSet) /  float (len (dataSet))
            # 计算熵值
            newEntropy += prob * calcShannonEnt (subDataSet)
        infoGain = baseEntropy - newEntropy
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
        
if __name__ == '__main__':
    dataSet, features = createDataSet()
    print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))
    