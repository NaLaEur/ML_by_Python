import os
import jieba

def TextProcessing(folder_path):
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
    print(data_list)
    print(class_list)
            


if __name__ == '__main__':
    #文本预处理
    folder_path = 'Sample'                #训练集存放地址
    TextProcessing(folder_path)