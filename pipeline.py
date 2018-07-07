import func
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import logistic
from sklearn.metrics import classification_report
from sklearn.model_selection import  GridSearchCV

train_label_list,train_corpus_list=func.part_jieba('data/train_set.txt')
test_label_list,test_corpus_list=func.part_jieba('data/test_set.txt')


##pipeline方法一般只使用sklearn中已经封装了的函数
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf',RandomForestClassifier())
])
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-2, 1e-3),
 }


#GridSearchCV用于寻找vectorizer词频统计, tfidftransformer特征变换和分类模型的最优参数
grid_search = GridSearchCV( text_clf, parameters)
print('grid_search','\n',grid_search,'\n') #输出所有参数名及参数候选值
grid_search.fit(train_corpus_list,train_label_list),'\n'#遍历执行候选参数，寻找最优参数



best_parameters = dict(grid_search.best_estimator_.get_params())#get实例中的最优参数
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name])),'\n'#输出参数结果


#将pipeline_obj实例中的参数重写为最优结果'''
text_clf.fit(train_corpus_list,train_label_list)
print(text_clf.score(test_corpus_list,test_label_list))

