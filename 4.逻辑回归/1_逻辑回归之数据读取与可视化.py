import numpy as np
import matplotlib.pyplot as plt

"""
函数说明：加载数据
Parameters:
    无
return:
    数据
    标签
"""
def LoadDataSet ():
    filename = "testSet.txt"
    data  = []
    label = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            linArr = line.strip().split()
            data.append ([1.0, float(linArr[0]), float(linArr[1])])
            label.append (int(linArr[2]))
    return data, label

"""
函数说明 数据可视化
Parameters:
    无
Return：
    图像可视化
"""    
def PlotDataSet ():
    # 1 获取数据,返回的数据是列表形式
    data, label = LoadDataSet()
    # 2 将数据转换为数组形式
    dataArr = np.array (data)
    # 3 获取现有数据一共有多少变量
    n = np.shape (dataArr)[0]
    # 4 定义数组，分别存储 0 和 1 的数据
    xcord1 = []; ycord1 = []                                            #正样本
    xcord2 = []; ycord2 = [] 
    for i in range(n):
        if int (label[i]) == 1:
            xcord1.append (dataArr[i, 1])
            ycord1.append (dataArr[i, 2])
        else:
            xcord2.append (dataArr[i, 1])
            ycord2.append (dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本
    plt.title('DataSet')                                                #绘制title
    plt.xlabel('x'); plt.ylabel('y')                                    #绘制label
    plt.show()         
    
if __name__ == "__main__":
    PlotDataSet()