# decompyle3 version 3.3.2
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar 25 2019, 22:22:05) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: /home/zchelllo/humor/code/baseline.py
# Size of source mod 2**32: 622 bytes
import pandas as pd, numpy as np
from util import util
from sklearn.metrics import accuracy_score

def baseline():
    A_train = pd.read_csv('./data/task1/train.csv')
    A_test = pd.read_csv('./data/task1/truth_task_1.csv')
    B_train = pd.read_csv('./data/task2/train.csv')
    B_test = pd.read_csv('./data/task2/truth_task_2.csv')
    pred = np.full(len(B_test['label']), B_train['label'].value_counts().idxmax())
    print('Subtask A baseline RMSE: %.5f' % util.rmse_A(A_test.meanGrade, np.mean(A_train.meanGrade)))
    print('Subtask B baseline accuracy: {}'.format(accuracy_score(B_test.label, pred)))