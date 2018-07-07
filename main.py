import func
##########引入写的函数############

#引入训练集和测试集（包括标签和句子）
train_label_list,train_corpus_list=func.part_jieba('data/train_set.txt')
test_label_list,test_corpus_list=func.part_jieba('data/test_set.txt')

#训练模型
train_tfidf,transformer,vectorizer=func.tfidf_cop(train_corpus_list)#计算训练集tfidf
classiy_model=func.model_train(train_tfidf,train_label_list) #训练模型并测试

#测试
test_tfidf=func.tfidf_cop2(test_corpus_list,transformer,vectorizer)#计算测试集tfidf
test1=test_tfidf[:1000]
func.model_test(test_tfidf,test_label_list,test1,classiy_model)