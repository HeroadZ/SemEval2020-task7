import pandas as pd
import numpy as np
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import os
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score

# user import
from util import util


class My_BERT:

    def __init__(self, pair=True, model_name='bert-base-cased', max_len=256, aug=False, clf=False):
        self.device = torch.device(
            'cuda:5' if torch.cuda.is_available() else 'cpu')
        self.CONFIG = {
            'A_train_p': './data/task1/train.csv',
            'A_dev_p': './data/task1/dev.csv',
            'A_test_p': './data/task1/truth_task_1.csv',
            'B_train_p': './data/task2/train.csv',
            'B_dev_p': './data/task2/dev.csv',
            'B_test_p': './data/task2/truth_task_2.csv',
        }
        self.pair = pair
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=1 if not clf else 3).to(self.device)
        self.max_len = max_len
        self.aug = aug
        self.clf = clf

    def fineT(self, epochs=1, bs=16, verbose=False):

        if verbose:
            print('you are now using {}.'.format(
            torch.cuda.get_device_name(self.device)))

        # read dataset
        if not self.aug and not self.clf:
            a_t_s, a_t_l = util.Read_data(
                self.CONFIG['A_train_p']).reader(task="A", pair=self.pair)
            a_v_s, a_v_l = util.Read_data(
                self.CONFIG['A_dev_p']).reader(task="A", pair=self.pair)
            s, l = a_t_s + a_v_s, a_t_l + a_v_l
        if self.aug:
            s, l = util.data_aug(os.getcwd() + '/data/aug.txt')

        if self.clf:
            b_t_s, b_t_l = util.Read_data(
                self.CONFIG['B_train_p']).reader(task="B", pair=False)
            b_v_s, b_v_l = util.Read_data(
                self.CONFIG['B_dev_p']).reader(task="B", pair=False)
            s, l = b_t_s + b_v_s, b_t_l + b_v_l
            t_ids, t_masks, t_types = (torch.tensor(x)
                                       for x in util.get_pair_ids(s, self.tokenizer, max_len=self.max_len, pair=True))
        else:
            t_ids, t_masks, t_types = (torch.tensor(x)
                           for x in util.get_pair_ids(s, self.tokenizer, max_len=self.max_len, pair=self.pair))
        # create dataloader

        train_data = TensorDataset(t_ids, t_masks, t_types, torch.tensor(l))
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=bs)

        # optimizer initial
        optimizer = AdamW(self.model.parameters(), lr=2e-05, eps=1e-08)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # start training
        loss_values = []
        self.model.zero_grad()
        for epoch_i in range(epochs):
            if verbose:
                print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

            t0 = time.time()
            total_loss = 0
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                if step % 100 == 0 and not step == 0:
                    elapsed = util.format_time(time.time() - t0)
                    if verbose:
                        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                            step, len(train_dataloader), elapsed))

                b_input_ids, b_input_mask, b_types, b_labels = tuple(
                    (t.to(self.device) for t in batch))
                outputs = self.model(input_ids=b_input_ids, token_type_ids=b_types,
                                attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

            avg_train_loss = total_loss / len(train_dataloader)
            loss_values.append(avg_train_loss)
            if verbose:
                print('  Average training loss: {0:.2f}'.format(
                    avg_train_loss))
                print('  Training epcoh took: {:}'.format(
                    util.format_time(time.time() - t0)))

        if verbose:
            print('Training Finished.')

        file_name = '/pretrained/{2}/Bert_{2}_regress_{0}epochs_{1}bs.pt'.format(
            epochs, bs, 'pair' if self.pair else 'single')
        if self.aug:
            file_name = '/pretrained/aug/Aug_{0}epochs_{1}bs.pt'.format(epochs, bs)
        if self.clf:
            file_name = '/pretrained/clf/Clf_{0}epochs_{1}bs.pt'.format(epochs, bs)

        torch.save(self.model.state_dict(), os.getcwd() + file_name)

        print(file_name + ' saved.')
        torch.cuda.empty_cache()
        return self

    def predict(self, pretrained=None, task='A', bs=128):
        test_s, test_l = util.Read_data(
            self.CONFIG['A_test_p'] if task == 'A' else self.CONFIG['B_test_p']).reader(task=task, pair=self.pair)

        if task != 'A' and not self.clf:
            test_s = [x[0] for x in test_s] + [x[1] for x in test_s]

        # create dataloader
        if self.clf:
            ids, masks, types = (torch.tensor(x) for x in util.get_pair_ids(
                test_s, self.tokenizer, max_len=self.max_len, pair=True))
        else:
            ids, masks, types = (torch.tensor(x) for x in util.get_pair_ids(
                test_s, self.tokenizer, max_len=self.max_len, pair=self.pair))
        
        data = TensorDataset(ids, masks, types)
        dataloader = DataLoader(data, batch_size=bs)

        tag = 'pair' if self.pair else 'single'
        if self.aug:
            tag = 'aug'
        if self.clf:
            tag = 'clf'
        if pretrained:
            self.model.load_state_dict(torch.load(
                    '{0}/pretrained/{1}/{2}'.format(os.getcwd(), tag, pretrained)))
        self.model.eval()

        pred = []
        for step, batch in enumerate(dataloader):
            b_input_ids, b_input_mask, b_types = tuple(
                (t.to(self.device) for t in batch))
            with torch.no_grad():
                outputs = self.model(
                    input_ids=b_input_ids, token_type_ids=b_types, attention_mask=b_input_mask)
            logits = outputs[0]
            # Move logits and labels to CPU
            pred.extend(logits.detach().cpu().numpy())
        
        return pred



    def evaluate(self, pretrained=None, task='A',  bs=128):
        pred = [util.make_even(x) for x in self.predict(pretrained=pretrained, task=task, bs=bs)]
        _, test_l = util.Read_data(
            self.CONFIG['A_test_p'] if task == 'A' else self.CONFIG['B_test_p']).reader(task=task, pair=self.pair)
        if task == 'A':
            print('{} model task A RMSE = {:.5f}'.format(pretrained[:-3] if pretrained else 'this model', util.rmse_A(test_l, pred)))
        else:
            if not self.clf:
                pred = util.to_label(pred, len(test_l))
            print('{} model task B accuracy = {:.5f}'.format(
                pretrained[:-3] if pretrained else 'this model', accuracy_score(test_l, pred)))
