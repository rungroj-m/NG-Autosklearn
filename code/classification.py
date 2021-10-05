from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pandas as pd
import csv
import autosklearn.classification
import datetime

class Classification:
    
    def __init__(self, project_name):

        now = datetime.datetime.now()
        time = '{:02d}'.format(now.day) + '{:02d}'.format(now.month) + str(now.year)
        self.filename = project_name + "_" + time + ".log"
            
        with open(self.filename, 'w') as f:
            f.write("create classification") 
            
    def set_n_gram(self, filename):
        n_grams = list()
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            for row in reader:
                n_grams.append(row[5].strip())
        self.n_grams = n_grams
        
    def vectorization(self, comment):
        lens = [len(x.split()) for x in self.n_grams]
        mn, mx = (min(lens), max(lens))
        self.vect = CountVectorizer(vocabulary=self.n_grams, ngram_range=(mn, mx))
        return self.vect.fit_transform(comment)
                                            
    def ten_fold(self,X,y):
#         sss = StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
        sss = StratifiedShuffleSplit(n_splits=10,random_state=1)
        self.pp(str(sss))
        runner = 0
        for train_index, test_index in sss.split(X,y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_test_class = (np.unique(y_test))
            automl = autosklearn.classification.AutoSklearnClassifier(
                memory_limit=1024*32, time_left_for_this_task = 1*60*60, metric=autosklearn.metrics.f1_weighted
            )

            automl.fit(X_train.copy(), y_train.copy())
            automl.refit(X_train.copy(), y_train.copy())
            y_hat = automl.predict(X_test)
            predict_proba = automl.predict_proba(X_test)
            
            if len(np.unique(y)) == 2: # binary class
                roc_auc = roc_auc_score(y_test,predict_proba[:,1])
            else: # multi class
                roc_auc = roc_auc_score(y_test, predict_proba ,average='weighted',multi_class='ovr',labels=y_test_class)
                
            self.pp("round: " + str(runner) + "Classification report" + str(classification_report(y_test, y_hat)))
            self.pp("round: " + str(runner) + "ROC AUC" + str(roc_auc))
            self.pp("round: " + str(runner) + "Confusion matrix" + str(confusion_matrix(y_test, y_hat)))
            self.pp("show_models: " + str(automl.show_models()))
            self.pp("sprint_statistics: " + str(automl.sprint_statistics()))
            runner += 1
            
    def important_feature(self,X,y):
        # for each class
        feature_len = X.shape[1]
        y_class = np.unique(y)
        word_list = self.vect.get_feature_names()

        result = dict()
        for number in y_class:
            interest_y = np.where(y == number)
            interest_x = X[interest_y]
            count_list = interest_x.sum(axis=0).tolist()[0]            
            word_count = dict((k, v) for k, v in dict(zip(word_list, count_list)).items() if v > 2)
            result[number] = dict(sorted(word_count.items(), key=lambda item: item[1], reverse = True))

        return result
    
    def pp(self,comment):        
        with open(self.filename,'a') as f:
            f.write(comment + '\n')            
        print(comment)