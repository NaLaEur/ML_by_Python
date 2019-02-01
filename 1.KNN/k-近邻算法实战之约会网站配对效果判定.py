import numpy as np
import operator
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
# 1 读取文件
def file2matrix (filename):
    # 1 打开文件
    fr = open (filename)
    # 2 读取文件内容
    arrayOLines = fr.readlines()
    # 3 得到文件行数
    numberOfLines = len (arrayOLines)
    #返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat = np.zeros ((numberOfLines, 3))
    # 返回的分类标签向量
    classLabelVector = []
    # 行的索引值
    index = 0
    
    for line in arrayOLines:
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip ()
        #使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == '1':
            classLabelVector.append(1)
        elif listFromLine[-1] == '2':
            classLabelVector.append(2)
        elif listFromLine[-1] == '3':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

# 2 对数据进行可视化处理
def showDataSet (datingDataMat, datingLabels):
    # 1 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    #2 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #  当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots (nrows = 2, ncols = 2, sharex = False, sharey = False, figsize = (13, 8))
    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占',FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')
 
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')
 
    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red') 
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black') 
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
    plt.tight_layout()
    plt.show()
 
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

# 对数据进行归一化操作
def autoNorm (dataSet):
    # 获取数据的最小值与最大值
    minVals = dataSet.min (0)
    maxVals = dataSet.max (0)
    # 获取最小值与最大值的范围
    ranges = maxVals - minVals
    # 创建一个尺寸大小与dataSet一致的矩阵
    normDataSet = np.zeros(np.shape(dataSet))
    # 获取行数，方便下面进行补充
    m = dataSet.shape[0]
    # 进行归一化操作
    #原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    #除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    
    return normDataSet, minVals, maxVals

def datingClassTest():
    #打开的文件名
    filename = "datingTestSet2.txt"
    #将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat, datingLabels = file2matrix(filename)
    #取所有数据的百分之十
    hoRatio = 0.10
    #数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #获得normMat的行数
    m = normMat.shape[0]
    #百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    #分类错误计数
    errorCount = 0.0
    showDataSet (datingDataMat, datingLabels)
    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],
            datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))
    
if __name__ == "__main__":
    datingClassTest()
    