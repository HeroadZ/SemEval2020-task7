from model.bert import My_BERT
from model.NB_SVM import NB_SVM
from model.bert_nbsvm import Bert_NBSVM
from util import util 
import torch

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
    # train_bert(pair=True, model='bert-base-cased', max_len=CONFIG['pair_max_len'])
    # for bert single regressor training
    # train_bert(pair=False, model='bert-base-cased', max_len=CONFIG['single_max_len'])
    # for bert single regressor + data augmentation training
    train_bert(pair=False, model='bert-base-cased', max_len=CONFIG['single_max_len'], aug=True)
    # for bert classifier
    train_bert(pair=False, model='bert-base-cased', max_len=CONFIG['pair_max_len'], clf=True)

def main():
    # for reproducing
    util.setup_seed(66)

    # util.baseline()

    train()

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
