from model.bert import My_BERT
from model.NB_SVM import NB_SVM
from util import util
from sklearn.metrics import accuracy_score

class Bert_NBSVM:

    def __init__(self, task='A', best_bert='Bert_pair_regress_1epochs_8bs.pt', bert_weight=0.91, nbsvm_weight=0.09):
        self.task = task
        self.CONFIG = {
            'pair_max_len': 160,
            'single_max_len': 96,  # max length for train, dev, test dataset is 74
            'A_test_p': './data/task1/truth_task_1.csv',
            'B_test_p': './data/task2/truth_task_2.csv',
        }
        self.best_bert = best_bert
        self.pair = True if 'pair' in best_bert else False
        self.bert_weight = bert_weight
        self.nbsvm_weight = nbsvm_weight
    
    def predict(self):
        nb_svm_pred = NB_SVM(task=self.task).train().predict()
        bert_pred = My_BERT(pair=self.pair,
                                 max_len=self.CONFIG['pair_max_len'] if self.pair else self.CONFIG['single_max_len']).predict(
                                     task=self.task, pretrained=self.best_bert)
        pred = [util.make_even(self.bert_weight*x + self.nbsvm_weight*y)
                for (x, y) in zip(bert_pred, nb_svm_pred)]
        if self.task != 'A':
            pred = util.to_label(pred, len(pred)//2)
        return pred
    
    def evaluate(self):
        pred = self.predict()
        _, test_l = util.Read_data(
            self.CONFIG['A_test_p'] if self.task == 'A' else self.CONFIG['B_test_p']).reader(task=self.task, pair=self.pair)
        if self.task == 'A':
            res = util.rmse_A(test_l, pred)
            print('{0}*{1} + {2}*nbsvm model task A RMSE = {3:.5f}'.format(
                self.bert_weight, self.best_bert[:-3], self.nbsvm_weight, res))
        else:
            res = accuracy_score(test_l, pred)
            print('{0}*{1} + {2}*nbsvm model task B accuracy = {3:.5f}'.format(
                self.bert_weight, self.best_bert[:-3], self.nbsvm_weight, res))
        return round(res, 5)
