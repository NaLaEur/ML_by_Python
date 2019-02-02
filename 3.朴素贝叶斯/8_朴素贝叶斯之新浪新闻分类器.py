import os
import jieba
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
"""
函数目的：
    读取数据，将数据分为训练集和测试集，并对数据中出现的词汇频率进行高低排序
paramters：
    folder_path - 文件地址
    test_size   - 测试集大小
 return:
    按照频率排序的数据
    训练集和测试集数据
By NaLaEur
    
"""
def TextProcessing(folder_path, test_size = 0.2):
    # 1 查看 folder_path下的文件
    folder_list = os.listdir (folder_path)
    # 2 训练集数据
    data_list  = []
    class_list = []
    
    # 3 遍历每个文件夹中的内容
    for folder in  folder_list:
        # 3.1 获取路径的新地址，以前的加现在的
        new_folder_path = os.path.join (folder_path, folder)
        # 3.2 获取文件夹中有哪些文件
        files = os.listdir (new_folder_path)
        # 3.3 打开文件夹中的所有文件，使用jieba对其进行切分，并将词加入数据list中
        for file in files:
            # 3.4 打开文件
            with open (os.path.join (new_folder_path, file), 'r', encoding = 'utf-8') as f: 
                raw = f.read ()
            # 3.5 使用jieba 对文件进行切分
            word_cut = jieba.cut(raw, cut_all = False)            #精简模式，返回一个可迭代的generator
            word_list = list(word_cut)
            # 3.6 加入元素
            data_list.append(word_list)
            class_list.append(folder)
    
    # 4 将原始数据切分为训练集和测试集
    # 4.1 将数据和列表压缩，组成一个组合
    data_class_list = list (zip (data_list, class_list))
    # 4.2 将数据进行打乱处理
    random.shuffle (data_class_list)
    # 4.3 获得训练集中一共有多少个数据
    index = int (len (data_class_list) * test_size) + 1
    # 4.4 切分训练集和测试集
    train_list = data_class_list[index:]
    test_list  = data_class_list[:index]
    # 4.5 对训练集和测试集进行切分
    train_data_list, train_class_list = zip (* train_list)
    test_data_list, test_class_list = zip (* test_list)
    
    # 5 声明一个字典，存储train_data_list中的所有单词
    all_words_dict = {}
    for world_list in train_data_list:
        for doc in world_list:
            if doc in all_words_dict.keys():
                all_words_dict[doc] += 1
            else:
                all_words_dict[doc] = 1
    
    #6 根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key = lambda f:f[1], reverse = True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)    #解压缩
    all_words_list = list(all_words_list)                        #转换成列表
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list
    
"""
函数说明:读取一份结束词汇文件里的内容
 
Parameters:
    words_file - 文件路径
Returns:
    words_set - 读取的内容的set集合
Author:
    NaLaEur
"""  
def MakeWordsSet(words_file):    
    words_set = set()
    # 只读形式打开文件
    with open(words_file, 'r', encoding = 'utf-8') as f: 
        # 逐行读取
        for line in f.readlines():
            word = line.strip()
            if (len (word) > 0):
                words_set.add (word)
    return words_set

"""
函数说明:文本特征选取
 
Parameters:
    all_words_list - 训练集所有文本列表
    deleteN - 删除词频最高的deleteN个词
    stopwords_set - 指定的结束语
Returns:
    feature_words - 特征集，词典
Author:
    NaLaeur
"""

def words_dict(all_words_list, deleteN, stopwords_set = set()):
    feature_words = []                            #特征列表
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:                            #feature_words的维度为1000
            break                               
        #如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1   
    return feature_words
"""
函数说明:根据feature_words将文本向量化，处理数据
 
Parameters:
    train_data_list - 训练集
    test_data_list - 测试集
    feature_words - 特征集
Returns:
    train_feature_list - 训练集向量化列表
    test_feature_list - 测试集向量化列表
Author:
    NaLaEur
"""
def TextFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):                        #出现在特征集中，则置1                                               
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list                #返回结果

def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy
 
if __name__ == '__main__':
    #文本预处理
    folder_path = 'Sample'                #训练集存放地址
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
 
    # 生成stopwords_set
    stopwords_file = 'stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)
 
 
    test_accuracy_list = []
    # 删掉高频次个数
    deleteNs = range(0, 1000, 20)                #0 20 40 60 ... 980
    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
        test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list.append(test_accuracy)
 
    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()
    
    text = train_data_list[0]
    print (text)
    text_words = set (text)
    print (text_words)