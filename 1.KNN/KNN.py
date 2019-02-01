import numpy as np
import operator
# 创建数据
def createDataSet ():
    group = np.array ([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['爱情片','爱情片','动作片','动作片']
    return group, labels 

# 建立KNN分类器
def classify0 (inX, dataSet, labels, k):
    """
    输入：
        inX     待检测样本
        dataSet 训练集数据
        labels  训练集标签
        k       选取样本个数
    """
    # 1 将待测样本的个数与训练集的个数相等
    dataSetSize = dataSet.shape[0]
    #  np.tile (a, b)  将a按照b的shape进行复制
    diffMat     = np.tile (inX, (dataSetSize, 1)) - dataSet
    # 2 开始计算待测点距离训练集的距离
    sqDiffMat   = diffMat ** 2
    sqDistances = sqDiffMat.sum (axis = 1)
    distances   = sqDistances ** 0.5
    # 3 获取每个编号
    sortedDistIndices = distances.argsort ()
    # 4 建立一个字典，将优先级较高的 k 个判断结果取出来，存入字典中
    classcout = {}
    for i in range (k):
        if labels[sortedDistIndices[i]] not in classcout.keys():
            classcout[labels[sortedDistIndices[i]]] = 1
        else:
            classcout[labels[sortedDistIndices[i]]] += 1
    # 5 对字典中的大小顺序进行排序
        # key=operator.itemgetter(1)根据字典的值进行排序
        # key=operator.itemgetter(0)根据字典的键进行排序
        # reverse降序排序字典        
    
    sortedClassCount = sorted(classcout.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
    
if __name__ == "__main__":
    group, labels = createDataSet()   
    print (group)
    #创建数据集
    group, labels = createDataSet()
    #测试集
    test = [101,20]
    #kNN分类
    test_class = classify0(test, group, labels, 3)
    #打印分类结果
    print(test_class)