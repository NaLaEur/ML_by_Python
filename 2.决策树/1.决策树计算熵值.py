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
    print (numEntries)
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
    
if __name__ == "__main__":
    dataSet, labels = createDataSet()
    shannonEnt = calcShannonEnt (dataSet)
    print (shannonEnt)
    