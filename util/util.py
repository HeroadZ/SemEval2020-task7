import json, pandas as pd, numpy as np, time, datetime, pickle, random, copy, torch, zipfile, re, string, math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

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
    print('Subtask A baseline RMSE: %.5f' % rmse_A(
        A_test.meanGrade, np.mean(A_train.meanGrade)))
    print('Subtask B baseline accuracy: {}'.format(
        accuracy_score(B_test.label, pred)))

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
