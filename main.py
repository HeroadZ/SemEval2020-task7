from model.bert import My_BERT
from model.NB_SVM import NB_SVM
from model.bert_nbsvm import Bert_NBSVM
from util import util 
import torch
import os
import numpy as np

CONFIG = {
    'pair_max_len': 160,
    'single_max_len': 96, # max length for train, dev, test dataset is 74
}


def train_bert(pair=False, model='bert-base-cased', max_len=CONFIG['single_max_len'], aug=False, clf=False):
    for i in range(1, 5):
            for j in [32, 16, 8, 4]:
                My_BERT(
                    pair=pair, model_name=model, max_len=max_len, aug=aug, clf=clf).fineT(epochs=i, bs=j, verbose=False)


def train():
    # training part, training file will be saved in pretrained folder
    # for bert pair regressor training
    train_bert(pair=True, model='bert-base-cased', max_len=CONFIG['pair_max_len'])
    # for bert single regressor training
    train_bert(pair=False, model='bert-base-cased', max_len=CONFIG['single_max_len'])
    # for bert single regressor + data augmentation training
    train_bert(pair=False, model='bert-base-cased', max_len=CONFIG['single_max_len'], aug=True)
    # for bert classifier
    train_bert(pair=False, model='bert-base-cased', max_len=CONFIG['pair_max_len'], clf=True)

def print_table(a_res=None, b_res=None):
    header = ['batch_size', '1epochs', '2epochs', '3epochs', '4epochs']
    column = ['bs=4', 'bs=8', 'bs=16', 'bs=32']
    row_format = "{:>15}" * len(header)
    if a_res is None and b_res is None:
        print('no data to show')
        return 
    if a_res is not None:
        print('task A RMSE results:')
        print(row_format.format(*header))
        for team, row in zip(column, a_res):
            print(row_format.format(team, *row))
    if b_res is not None:
        print('task B accuracy results:')
        print(row_format.format(*header))
        for team, row in zip(column, b_res):
            print(row_format.format(team, *row))

def main():
    # for reproducing
    util.setup_seed(66)

    # util.baseline()

    # train()

    # My_BERT(
    #     pair=False, max_len=CONFIG['pair_max_len'], clf=True).evaluate(pretrained='Clf_1epochs_4bs.pt', task='B')
    # print(os.listdir(os.getcwd()+'/pretrained/'))
    print_table(a_res=np.full((4, 4), 0.51), b_res=np.full((4, 4), 0.63))

    # My_BERT(
    #     pair=True, model_name='bert-base-cased', max_len=CONFIG['pair_max_len']).fineT(epochs=1, bs=128, verbose=True).evaluate(
    #         task='A'
    #     )

    # My_BERT(pair=True, model_name='bert-base-cased', max_len=CONFIG['pair_max_len']).evaluate(
    #     task='A', pretrained='Bert_pair_regress_1epochs_32bs.pt', bs=128)
    
 
    # nb_svm = NB_SVM(task='A')
    # nb_svm.train().evalute()

    # Bert_NBSVM(task='B', best_bert='Bert_pair_regress_1epochs_8bs.pt').evaluate()

    pass

if __name__ == '__main__':
    main()
