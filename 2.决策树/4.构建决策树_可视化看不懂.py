from math import log
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import operator
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
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']             #分类属性
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
   #     print("第%d个特征的增益为%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

"""
函数说明：创建决策树

Paramters:
    dataSet - 训练数据集
    labels  - 分类属性标签
    featLabels - 存储选择的最优特征标签
return :
    myTree - 决策树
"""

def majorityCnt(classList):
    classCount = {}
    for vote in classList:                                        #统计classList中每个元素出现的次数
        if vote not in classCount.keys():classCount[vote] = 0   
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)        #根据字典的值降序排序
    return sortedClassCount[0][0]                                #返回classList中出现次数最多的元素
 

def createTree(dataSet, labels, featLabels):
    # 获取所有标签
    classList = [example[-1] for example in dataSet]
    # 如果现在classList中只有一个元素的时候，返回classList中的元素
    # 如果类别相同，就停止继续划分
    if classList.count (classList[0]) == len (classList):
        return classList[0]
    # 遍历完所有特征时返回出现最多的类的标签
    if len(dataSet[0]) == 1 or len(labels) == 0:
        return majorityCnt (classList)
    # 1 选择最优特征标签
    bestFeat = chooseBestFeatureToSplit (dataSet)
    print (labels[bestFeat])
    # 2 找到最优特征的标签
    bestFeatLabel = labels[bestFeat]
    # 3 将获得的标签放在特征Labels中
    featLabels.append (bestFeatLabel)
    # 4 根据最优特征的标签生成树
    myTree = {bestFeatLabel:{}}
    # 5 删除已经使用特征标签
    del(labels[bestFeat])
    # 6 得到训练集中所有最优特征的属性值
    featValues = [example[bestFeat] for example in dataSet]
   
    # 7 去掉重复的属性值
    uniqueVals = set(featValues)
    # 8 遍历特征，创建决策树。 
    for value in uniqueVals:                                    #删掉那一项，找到下一个最大信息增益                       
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree

"""
函数说明:获取决策树叶子结点的数目
 
Parameters:
    myTree - 决策树
Returns:
    numLeafs - 决策树的叶子结点的数目
"""
def getNumLeafs(myTree):
    numLeafs = 0                                                #初始化叶子
    firstStr = next(iter(myTree))                                #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                #获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs
 
"""
函数说明:获取决策树的层数
 
Parameters:
    myTree - 决策树
Returns:
    maxDepth - 决策树的层数
"""
def getTreeDepth(myTree):
    maxDepth = 0                                                #初始化决策树深度
    firstStr = next(iter(myTree))                                #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]                                #获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth            #更新层数
    return maxDepth
 
"""
函数说明:绘制结点
 
Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
Returns:
    无
"""
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")                                            #定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)        #设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',    #绘制结点
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)
 
"""
函数说明:标注有向边属性值
 
Parameters:
    cntrPt、parentPt - 用于计算标注位置
    txtString - 标注的内容
Returns:
    无
"""
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]                                            #计算标注位置                   
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
 
"""
函数说明:绘制决策树
 
Parameters:
    myTree - 决策树(字典)
    parentPt - 标注的内容
    nodeTxt - 结点名
Returns:
    无
"""

def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")                                        #设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")                                            #设置叶结点格式
    numLeafs = getNumLeafs(myTree)                                                          #获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)                                                            #获取决策树层数
    firstStr = next(iter(myTree))                                                            #下个字典                                                 
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)    #中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)                                                    #标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)                                        #绘制结点
    secondDict = myTree[firstStr]                                                            #下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD                                        #y偏移
    for key in secondDict.keys():                               
        if type(secondDict[key]).__name__=='dict':                                            #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))                                        #不是叶结点，递归调用继续绘制
        else:                                                                                #如果是叶结点，绘制叶结点，并标注有向边属性值                                             
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
    

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')                                                    #创建fig
    fig.clf()                                                                                #清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)                                #去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))                                            #获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))                                            #获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;                                #x偏移
    plotTree(inTree, (0.5,1.0), '')                                                            #绘制决策树
    plt.show()              
    
if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    print(myTree)
    createPlot(myTree)