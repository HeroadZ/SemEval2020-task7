import re, string
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from util import util
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class NB_SVM:

    def __init__(self, task='A'):
        self.CONFIG = {
            'A_train_p': './data/task1/train.csv',
            'A_dev_p': './data/task1/dev.csv',
            'A_test_p': './data/task1/truth_task_1.csv',
            'B_train_p': './data/task2/train.csv',
            'B_dev_p': './data/task2/dev.csv',
            'B_test_p': './data/task2/truth_task_2.csv',
        }
        self.task = task
        self.retok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
        self.model = SVR(C=1.0, epsilon=0.3)
        self.tv_f = None
        self.tv_l = None
        self.test_f = None
        self.test_l = None

    def get_data(self):
        t_s, t_l = util.Read_data(
            self.CONFIG['A_train_p']).reader(task='A', pair=False)
        v_s, v_l = util.Read_data(
            self.CONFIG['A_dev_p']).reader(task='A', pair=False)
        tv_s, tv_l = t_s + v_s, t_l + v_l

        test_s, test_l = util.Read_data(self.CONFIG['A_test_p'] if self.task == 'A' else self.CONFIG['B_test_p']).reader(
            task=self.task, pair=False)

        if self.task != 'A':
            test_s = [x[0] for x in test_s] + [x[1] for x in test_s]
        return (tv_s, tv_l, test_s, test_l)

    def tokenize(self, s):
        return self.retok.sub(r' \1 ', s).split()

    def pr(self, trn_term_doc, y_i, y):
        p = trn_term_doc[(y == y_i)].sum(0)
        return (p + 1) / ((y == y_i).sum() + 1)

    def get_features(self):
        tv_s, tv_l, test_s, test_l = self.get_data()
        vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=(self.tokenize), min_df=3,
                              max_df=0.9,
                              strip_accents='unicode',
                              use_idf=1,
                              smooth_idf=1,
                              sublinear_tf=1)
        trn_term_doc = vec.fit_transform(tv_s)
        test_term_doc = vec.transform(test_s)
        r = np.log(self.pr(trn_term_doc, 1, np.array(tv_l)) /
                   self.pr(trn_term_doc, 0, np.array(tv_l)))
        tv_f = trn_term_doc.multiply(r)
        test_f = test_term_doc.multiply(r)
        return (tv_f, tv_l, test_f, test_l)

    def train(self):
        self.tv_f, self.tv_l, self.test_f, self.test_l = self.get_features()
        self.model.fit(self.tv_f, self.tv_l)
        return self

    def predict(self):
        return self.model.predict(self.test_f)


    def evalute(self):
        if self.task == 'A':
            print('NB-SVM model task A RMSE: %.5f' % util.rmse_A(self.test_l, self.predict()))
        else:
            print('NB-SVM model task B accuracy: %.5f' %
                  accuracy_score(self.test_l, util.to_label(self.predict(), len(self.test_l))))
