import json, pandas as pd, numpy as np, time, datetime, pickle, random, copy, torch, zipfile, re, string, math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

class Read_data:

    def __init__(self, train_path, dev_path, test_path):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

    def replace_edit(self, o, e, both=True):
        assert len(o) == len(e)
        s_o, s_e = [], []
        for i in range(len(o)):
            s_o.append(re.sub('<(.+)/>', '\\g<1>', o[i]))
            s_e.append(re.sub('<.+/>', e[i], o[i]))

        if both:
            return list(zip(s_o, s_e))
        return s_e

    def A_helper(self, df):
        sents = self.replace_edit(df['original'], df['edit'])
        try:
            labels = df['meanGrade'].tolist()
            return (
             sents, labels)
        except KeyError:
            return sents

    def A_reader(self):
        t_s, t_l = self.A_helper(pd.read_csv(self.train_path))
        v_s, v_l = self.A_helper(pd.read_csv(self.dev_path))
        test_s, test_l = self.A_helper(pd.read_csv(self.test_path))
        return (
         t_s, t_l, v_s, v_l, test_s, test_l)

    def B_helper(self, df):
        sents = list(zip(self.replace_edit(df['original1'], df['edit1']), self.replace_edit(df['original2'], df['edit2'])))
        try:
            labels = df['label'].tolist()
            return (
             sents, labels)
        except KeyError:
            return sents

    def B_reader(self):
        t_s, t_l = self.B_helper(pd.read_csv(self.train_path))
        v_s, v_l = self.B_helper(pd.read_csv(self.dev_path))
        test_s, test_l = self.B_helper(pd.read_csv(self.test_path))
        return (
         t_s, t_l, v_s, v_l, test_s, test_l)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_pair_ids(sents, tokenizer, max_len=256):
    input_ids, attention_masks, type_ids = [], [], []
    for e in sents:
        t_e = tokenizer.encode_plus(text=(e[0]), text_pair=(e[1]),
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


class NB_SVM:
    tv_f = tv_l = test_f = test_l = None
    model = SVR(C=1.0, epsilon=0.3)
    re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")

    def __init__(self, train_path, dev_path, test_path, task='A'):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.task = task

    def replace_edit(self, o, e, both=True):
        assert len(o) == len(e)
        s_o, s_e = [], []
        for i in range(len(o)):
            s_o.append(re.sub('<(.+)/>', '\\g<1>', o[i]))
            s_e.append(re.sub('<.+/>', e[i], o[i]))

        if both:
            return list(zip(s_o, s_e))
        return s_e

    def read_data(self):
        tv = pd.concat([pd.read_csv(self.train_path), pd.read_csv(self.dev_path)])
        tv_s = self.replace_edit((tv['original'].tolist()), (tv['edit'].tolist()), both=False)
        tv_l = tv['meanGrade'].values
        print(len(tv_s), len(tv_l))
        test = pd.read_csv(self.test_path)
        if self.task == 'A':
            test_s = self.replace_edit((test['original']), (test['edit']), both=False)
            test_l = test['meanGrade'].tolist()
        else:
            test_s = self.replace_edit((test['original1']), (test['edit1']), both=False) + self.replace_edit((test['original2']), (test['edit2']), both=False)
            test_l = test['label'].tolist()
        return (tv_s, tv_l, test_s, test_l)

    def tokenize(self, s):
        return NB_SVM.re_tok.sub(' \\1 ', s).split()

    def pr(self, trn_term_doc, y_i, y):
        p = trn_term_doc[(y == y_i)].sum(0)
        return (p + 1) / ((y == y_i).sum() + 1)

    def get_features(self):
        tv_s, tv_l, test_s, test_l = self.read_data()
        vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=(self.tokenize), min_df=3,
          max_df=0.9,
          strip_accents='unicode',
          use_idf=1,
          smooth_idf=1,
          sublinear_tf=1)
        trn_term_doc = vec.fit_transform(tv_s)
        test_term_doc = vec.transform(test_s)
        r = np.log(self.pr(trn_term_doc, 1, tv_l) / self.pr(trn_term_doc, 0, tv_l))
        tv_f = trn_term_doc.multiply(r)
        test_f = test_term_doc.multiply(r)
        return (
         tv_f, tv_l, test_f, test_l)

    def train(self):
        NB_SVM.tv_f, NB_SVM.tv_l, NB_SVM.test_f, NB_SVM.test_l = self.get_features()
        NB_SVM.model.fit(NB_SVM.tv_f, NB_SVM.tv_l)

    def predict(self):
        if self.task == 'A':
            return NB_SVM.model.predict(NB_SVM.test_f)
        pred = [make_even(x) for x in NB_SVM.model.predict(NB_SVM.test_f)]
        N = len(pred)
        return list(zip(pred[:N // 2], pred[N // 2:]))

    def evalute(self):
        if self.task == 'A':
            print('RMSE = %.3f' % rmse_A(NB_SVM.test_l, self.predict()))
        else:
            print('NB-SVM model task B accuracy: %.3f' % accuracy_score(NB_SVM.test_l, np.argmax((self.predict()), axis=1)))