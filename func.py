#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
头文件
'''
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


'''
函数stopwordslist
加载停用词表，产生停用词组成的列表
m为结果
'''
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
m=stopwordslist('data\stopwords.txt')




'''
提取文本中的文字
其中标签存在于label_list,内容存在corpus_list
参数：mypath 数据存储路径
     llist  标签列表
     corlist句子列表
'''
def part_jieba(mypath):
##################方法1，使用jieba分词#######################
    label_list=[] #用来存标签
    corpus_list=[] #用来存句子
    with open(mypath,encoding='utf-8') as f:
        for line in f:
            label_list.append(line.split('\t')[0])#tab前的数字标签数字部分
            data=line.split('\t')[1]#tab后的数据部分
            second_data=[x for x in jieba.cut(data)]#用jieba分割句子
            #print(ddata)
            # for i in range(len(ddata)): #去停用词
            #     if ddata[i] not in m:
            #         dddata+=ddata[i]+' '
            third_data=[second_data[i] for i in range(len(second_data)) if second_data[i] not in m]
            corpus_list.append(str(third_data))#讲每个句子存入list中
    return label_list,corpus_list
    print(label_list)
######################方法2：使用pyltp分词#################
def part_pyltp(mypath):
    from pyltp import Segmentor
    segmentor=Segmentor()#定义分词函数
    segmentor.load()

    with open(mypath) as f:
        for line in f:
            label_list.append(line.split('\t')[0])#tab前的数字标签数字部分
            data=line.split('\t')[1]#tab后的数据部分
            second_data=[x for x in list(segmentor.segment(data))]#用jieba分割句子
            #print(ddata)
            third_data=''#这里使用字符串存储每个句子是为了方便后面tfidf的得出
            for i in range(len(second_data)): #去停用词
                if second_data[i] not in m:
                    third_data+=second_data[i]+' '

            corpus_list.append(str(third_data))#讲每个句子存入list中




'''
tfidf
用于计算词的重要程度
这里使用了sklearn自带的方法
参数：corlist 处理好的句子
'''

def tfidf_cop(corlist):
    vectorizer=CountVectorizer(min_df=1e-5) #构造存储句子的容器，并从中选择大于1e-5的
    transformer=TfidfTransformer() #计算方法
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corlist)) #*****这里的结果不是list
    word=vectorizer.get_feature_names() #获取提取处的总的词
    return tfidf,transformer,vectorizer



def tfidf_cop2(corlist,transformer,vectorizer):
    tfidf = transformer.transform(vectorizer.transform(corlist))  # *****这里的结果不是list
    word = vectorizer.get_feature_names()  # 获取提取处的总的词
    return tfidf


'''
训练模型阶段
这里采取的是两种方法
分别为svc和randomforest
参数： tfidf 处理后的tfidf值
      label_list 标签
'''
def model_train(tfidf,label_list):
    classify_model=RandomForestClassifier()
#classify_model=SVC()
    classify_model.fit(tfidf,label_list)
    print('训练完成')
    return classify_model



'''
测试模型阶段
这里采用多种度量方法
参数： test_tfidf 测试用的tfidf
      test_label_list 测试用的标签
      test1 测试输出用的tfidf
      classify_model 训练好的模型 
'''
def model_test(test_tfidf,test_label_list,test1,classify_model):
    f1 = open('data\hey.txt', 'a')  # 用来在后面存储测试数据
    print( "val mean accuracy: {0}".format(classify_model.score(test_tfidf, test_label_list)))
    print(classification_report(test_label_list,classify_model.predict(test_tfidf)))
    y_pred=classify_model.predict(test1)
    print(y_pred)
    for i in range(len(y_pred)):
        f1.write(y_pred[i]+'\n')

#################使用管道方法，简化操作########################
# from sklearn.svm import SVC
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.pipeline import Pipeline
# text_clf=Pipeline([('vect',CountVectorizer()),
#                    ('tfidf',TfidfTransformer()),
#                    ('clf',SVC)])
#
# text_clf.fit(corpus_list,label_list)


