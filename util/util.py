import json, pandas as pd, numpy as np, time, datetime, pickle, random, copy, torch, zipfile, re, string, math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import os
from model.bert import My_BERT
from model.NB_SVM import NB_SVM
from model.bert_nbsvm import Bert_NBSVM
import pickle


CONFIG = {
    'pair_max_len': 160,
    'single_max_len': 96,  # max length for train, dev, test dataset is 74
}


class Read_data:

    def __init__(self, path):
        self.path = path

    def replace_edit(self, o, e, pair=True):
        assert len(o) == len(e)
        s_o, s_e = [], []
        for i in range(len(o)):
            s_o.append(re.sub('<(.+)/>', '\\g<1>', o[i]))
            s_e.append(re.sub('<.+/>', e[i], o[i]))
        if pair:
            return list(zip(s_o, s_e))
        return s_e

    def A_helper(self, df, pair=True):
        sents = self.replace_edit(df['original'], df['edit'], pair=pair)
        try:
            labels = df['meanGrade'].tolist()
            return (sents, labels)
        except KeyError:
            return sents

    def B_helper(self, df, pair=True):
        sents = list(zip(self.replace_edit(df['original1'], df['edit1'], pair=pair), self.replace_edit(df['original2'], df['edit2'], pair=pair)))
        try:
            labels = df['label'].tolist()
            return (sents, labels)
        except KeyError:
            return sents


    def reader(self, task='A', pair=True):
        if task == 'A':
            return self.A_helper(pd.read_csv(self.path), pair=pair)
        else:
            return self.B_helper(pd.read_csv(self.path), pair=pair)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def setup_seed(seed):
    CUDA = torch.cuda.is_available()
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_pair_ids(sents, tokenizer, max_len=300, pair=True):
    input_ids, attention_masks, type_ids = [], [], []
    for e in sents:
        if pair:
            t_e = tokenizer.encode_plus(text=(e[0]), text_pair=(e[1]),
            max_length=max_len,
          add_special_tokens=True,
          pad_to_max_length='right')
        else:
            t_e = tokenizer.encode_plus(text=e, 
                                        max_length=max_len,
                                        add_special_tokens=True,
                                        pad_to_max_length='right')
        input_ids.append(t_e['input_ids'])
        attention_masks.append(t_e['attention_mask'])
        type_ids.append(t_e['token_type_ids'])

    return (input_ids, attention_masks, type_ids)


def rmse_A(label, pred):
    rmse = np.sqrt(np.mean((np.array(label) - np.array(pred)) ** 2))
    return rmse


def to_csv(path, df, pred):
    output = pd.DataFrame({'id':df['id'],  'pred':pred})
    output.to_csv(path, index=False)
    with zipfile.ZipFile(path[:-4] + '.zip', 'w') as zf:
        zf.write(path)


def make_even(num):
    arr = np.linspace(0, 3, 16)
    diff, ans = (3, -1)
    for e in arr:
        if abs(e - num) < diff:
            diff = abs(e - num)
            ans = e

    return round(ans, 1)

def to_label(arr, N):
    pred_label = []
    for i in range(N):
        if arr[i] == arr[N+i]:
            pred_label.append(0)
        elif arr[i] > arr[N+i]:
            pred_label.append(1)
        else:
            pred_label.append(2)
    assert len(pred_label) == N
    return pred_label


def baseline():
    A_train = pd.read_csv('./data/task1/train.csv')
    A_test = pd.read_csv('./data/task1/truth_task_1.csv')
    B_train = pd.read_csv('./data/task2/train.csv')
    B_test = pd.read_csv('./data/task2/truth_task_2.csv')
    pred = np.full(len(B_test['label']),
                   B_train['label'].value_counts().idxmax())
    base_A = rmse_A(A_test.meanGrade, np.mean(A_train.meanGrade))
    base_B = accuracy_score(B_test.label, pred)
    return round(base_A, 5), round(base_B, 5)

def data_aug(path):
    s, l = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            s.append(line[4:])
            l.append(round(float(line[:3]), 1))
    return s, l

def gpu_info():
    if torch.cuda.device_count() > 0:
        for i in range(torch.cuda.device_count()):
            print('cuda {}: {}'.format(i, torch.cuda.get_device_name(i)))


def print_table(res=None, task='A'):
    if res is None:
        print('no data to show')
        return
    header = ['epochs', 'bs=32', 'bs=16', 'bs=8', 'bs=4']
    column = ['1epochs','2epochs', '3epochs', '4epochs']
    row_format = "{:>15}" * len(header)
    info = 'A RMSE' if task == 'A' else 'B accuracy'
    print('task {0} results:'.format(info))
    print(row_format.format(*header))
    N = len(column)
    for i in range(N):
        print(row_format.format(column[i], *[res[j]
                                             for j in range(i*N, (i+1)*N)]))


def get_res(tag, pair=True, max_len=CONFIG['pair_max_len'], task='A', aug=False, clf=False):
    # for pair regress
    if not os.path.exists(os.getcwd() +'/pretrained'):
        raise ValueError('no pretrained model files, please train first!')
    pretrains = os.listdir(os.getcwd()+'/pretrained/' + tag)
    model = My_BERT(pair=pair, max_len=max_len, aug=aug, clf=clf)
    res = [model.evaluate(pretrained=p, task=task) for p in pretrains]
    del model
    return res


def get_A():
    return {
        'pair_regress': get_res('pair', pair=True, max_len=CONFIG['pair_max_len'], task='A'),
        'single_regress': get_res('single', pair=False, max_len=CONFIG['single_max_len'], task='A'),
        'single+aug': get_res('aug', pair=False, max_len=CONFIG['single_max_len'], task='A', aug=True),
        'nbsvm': NB_SVM(task='A').train().evalute(),
        'pair+nbsvm': Bert_NBSVM(task='A', best_bert='Bert_pair_regress_1epochs_4bs.pt').evaluate(),
    }


def get_B():
    return {
        'pair_regress': get_res('pair', pair=True, max_len=CONFIG['pair_max_len'], task='B'),
        'single_regress': get_res('single', pair=False, max_len=CONFIG['single_max_len'], task='B'),
        'single+aug': get_res('aug', pair=False, max_len=CONFIG['single_max_len'], task='B', aug=True),
        'nbsvm': NB_SVM(task='B').train().evalute(),
        'pair+nbsvm': Bert_NBSVM(task='B', best_bert='Bert_pair_regress_1epochs_4bs.pt').evaluate(),
        'clf': get_res('clf', pair=False, max_len=CONFIG['pair_max_len'], task='B', clf=True),
    }


def save_res():
    res = {
        'A': get_A(),
        'B': get_B(),
    }
    with open('results.pickle', 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    return res

def show_A_res():
    if os.path.isfile('results.pickle'):
        with open('results.pickle', 'rb') as f:
            res = pickle.load(f)
    else:
        res = save_res()
    
    print('results for task A:')
    print('baseline: {}'.format(baseline()[0]))
    print('BERT pair:')
    print_table(res['A']['pair_regress'], task='A')
    print('BERT single:')
    print_table(res['A']['single_regress'], task='A')
    print('BERT single aug:')
    print_table(res['A']['single+aug'], task='A')
    print('NBSVM: {0}'.format(res['A']['nbsvm']))
    print('best BERT pair + NBSVM: {0}'.format(res['A']['pair+nbsvm']))

    return res['A']


def show_B_res():
    if os.path.isfile('results.pickle'):
        with open('results.pickle', 'rb') as f:
            res = pickle.load(f)
    else:
        res = save_res()

    print('results for task B:')
    print('baseline: {}'.format(baseline()[1]))
    print('BERT pair:')
    print_table(res['B']['pair_regress'], task='B')
    print('BERT single:')
    print_table(res['B']['single_regress'], task='B')
    print('BERT single aug:')
    print_table(res['B']['single+aug'], task='B')
    print('NBSVM: {0}'.format(res['B']['nbsvm']))
    print('best BERT pair + NBSVM: {0}'.format(res['B']['pair+nbsvm']))
    print('BERT pair classfier:')
    print_table(res['B']['clf'], task='B')
    return res['B']

def show_res():
    show_A_res()
    show_B_res()

def train_bert(pair=False, model='bert-base-cased', max_len=CONFIG['single_max_len'], aug=False, clf=False):
    for i in range(1, 5):
            for j in [32, 16, 8, 4]:
                My_BERT(
                    pair=pair, model_name=model, max_len=max_len, aug=aug, clf=clf).fineT(epochs=i, bs=j, verbose=False)


def train():
    # training part, training file will be saved in pretrained folder
    # for bert pair regressor training
    train_bert(pair=True, max_len=CONFIG['pair_max_len'])
    # for bert single regressor training
    train_bert(pair=False, max_len=CONFIG['single_max_len'])
    # for bert single regressor + data augmentation training
    train_bert(pair=False, max_len=CONFIG['single_max_len'], aug=True)
    # for bert classifier
    train_bert(pair=False, max_len=CONFIG['pair_max_len'], clf=True)


def calculate_acc_for_each_class(pred, label):
    N0, N1, N2 = label.count(0), label.count(1), label.count(2)
    dic = {
        0: 0,
        1: 0,
        2: 0,
    }
    for l, p in zip(label, pred):
        if l == p:
            dic[l] += 1
    print("label 0: {}, label 1: {}, label 2: {}".format(
        dic[0]/N0, dic[1]/N1, dic[2]/N2))
