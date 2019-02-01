from sklearn import tree
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    # 1 打开文件，读取数据
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    
    # 2 提取每组数据的类别，保存在列表里
    lenses_target = []                                                        
    for each in lenses:
        lenses_target.append(each[-1])
    
    # 3 将数据转换为pandas结构
    lenses_list = []
    lenses_dict = {}
    
    # 4 采取循环，一次遍历每一个特征
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append (each[lensesLabels.index (each_label)])
        lenses_dict[each_label] = lenses_list
        # 清空 list
        lenses_list = []
    # 转换为 pandas
    lenses_pd = pd.DataFrame(lenses_dict)                                    #生成pandas.DataFrame
    print(lenses_pd)  
    # 转为one-hot编码
    le = LabelEncoder()                                                        #创建LabelEncoder()对象，用于序列化            
    for col in lenses_pd.columns:                                            #为每一列序列化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)
    
    print ("The selsect of result:" , set(lenses_target))
    print (lenses_pd.columns)
    # 训练决策树
    clf = tree.DecisionTreeClassifier(max_depth = 4)                        #创建DecisionTreeClassifier()类
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)                    #使用数据，构建决策树
    # 预测结果
    print(clf.predict([[1,1,1,0]]))   
    