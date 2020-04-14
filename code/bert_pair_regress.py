import json, pandas as pd, numpy as np, time, datetime, pickle, torch
from torch.utils.data import TensorDataset, DataLoader
import random
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import copy, zipfile, re, string, math, sys, os
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from util import util
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('you are now using {}.'.format(torch.cuda.get_device_name(device)))
CONFIG = {'model_name':'bert-base-cased', 
 'file_path':{'A_train_p':'./data/task1/train.csv', 
  'A_dev_p':'./data/task1/dev.csv', 
  'A_test_p':'./data/task1/truth_task_1.csv', 
  'B_train_p':'./data/task2/train.csv', 
  'B_dev_p':'./data/task2/dev.csv', 
  'B_test_p':'./data/task2/truth_task_2.csv'}}

def fineT(epochs=1, bs=16, max_len=256, verbose=False):
    util.setup_seed(42)
    a_t_s, a_t_l, a_v_s, a_v_l, a_test_s, a_test_l = util.Read_data(CONFIG['file_path']['A_train_p'], CONFIG['file_path']['A_dev_p'], CONFIG['file_path']['A_test_p']).A_reader()
    b_t_s, b_t_l, b_v_s, b_v_l, b_test_s, b_test_l = util.Read_data(CONFIG['file_path']['B_train_p'], CONFIG['file_path']['B_dev_p'], CONFIG['file_path']['B_test_p']).B_reader()
    s, l = a_t_s + a_v_s, a_t_l + a_v_l
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    model = BertForSequenceClassification.from_pretrained((CONFIG['model_name']), num_labels=1)
    model.to(device)
    t_ids, t_masks, t_types = (torch.tensor(x) for x in util.get_pair_ids(s, tokenizer))
    train_data = TensorDataset(t_ids, t_masks, t_types, torch.tensor(l))
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=bs)
    optimizer = AdamW((model.parameters()), lr=2e-05,
      eps=1e-08)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
      num_training_steps=total_steps)
    loss_values = []
    model.zero_grad()
    for epoch_i in range(epochs):
        if verbose:
            print('')
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        else:
            t0 = time.time()
            total_loss = 0
            model.train()
            for step, batch in enumerate(train_dataloader):
                if step % 100 == 0:
                    if not step == 0:
                        elapsed = util.format_time(time.time() - t0)
                        if verbose:
                            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                    b_input_ids, b_input_mask, b_types, b_labels = tuple((t.to(device) for t in batch))
                    outputs = model(input_ids=b_input_ids, token_type_ids=b_types, attention_mask=b_input_mask, labels=b_labels)
                    loss = outputs[0]
                    total_loss += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

            avg_train_loss = total_loss / len(train_dataloader)
            loss_values.append(avg_train_loss)
        if verbose:
            print('  Average training loss: {0:.2f}'.format(avg_train_loss))
            print('  Training epcoh took: {:}'.format(util.format_time(time.time() - t0)))

    if verbose:
        print('Training Finished.')
    torch.save(model.state_dict(), os.getcwd() + '/pretrained/Bert_pair_regress_{0}epochs_{1}bs.pt'.format(epochs, bs))
    print('Bert_pair_regress_{0}epochs_{1}bs.pt saved'.format(epochs, bs))