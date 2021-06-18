import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import sys
from gensim.models import KeyedVectors
import nltk
from nltk.corpus import stopwords
# import jieba
import numpy as np
# import logging
import _public as pb
# from progressbar import *
from tqdm import tqdm
# from time import sleep
# import time
# from bert_serving.client import BertClient
import random
import math
import copy
import scipy.stats as stats
import json
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
from fairseq.models.roberta import RobertaModel



class SuperNetwork(nn.Module):
    def __init__(self):
        super(SuperNetwork, self).__init__()
        pass

    def forward(self, xs):
        pass

    def Train(self, train_loader, dev_loader, test_loader):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        if pb.Use_GPU == True:
            criterion = criterion.cuda()

        if pb.Use_GPU == True: torch.cuda.empty_cache()
        for i in range(self.epoch):
            if i == 0: self.Evaluate(train_loader, dev_loader, test_loader, i)
            self.train()
            trainbar, true_labels, factual_labels = tqdm(total=len(train_loader), ncols=pb.Tqdm_Len), [], []
            for batch_idx, (x, fcx, pcx, y, x_tensor, fcx_tensor, pcx_tensor, y_tensor) in enumerate(train_loader):
                optimizer.zero_grad()
                factual_output = self.forward(x_tensor)
                if pb.Weight==False:
                    loss = criterion(factual_output, y_tensor)
                else:
                    # pb.Print('pb.Weight=Ture', color='blue')
                    for j in range(len(factual_output)):
                        ps = F.softmax(factual_output[j])
                        index = y_tensor[j]
                        p = ps[index]
                        q = pb.Train_Distribution[index]
                        w = q / p
                        W = w if j == 0 else W + w
                    loss = None
                    for j in range(len(factual_output)):
                        ps = F.softmax(factual_output[j])
                        index = y_tensor[j]
                        p = ps[index]
                        l = -torch.log(p)
                        q = pb.Train_Distribution[index]
                        w = q / p
                        l = l * (w / W)
                        loss = l if j == 0 else loss + l
                    loss /= len(x)
                loss.backward()
                optimizer.step()
                if pb.Use_GPU == True: torch.cuda.empty_cache()
                trainbar.set_description(pb.Dataset_Name + ' Training_Epoch={}'.format(i+1))
                trainbar.update(1)
            trainbar.close()
            self.Evaluate(train_loader, dev_loader, test_loader, i+1)

    def Get_Counterfactual_Input(self, loader):
        if pb.Use_GPU == True: torch.cuda.empty_cache()

        matrix, cnt = torch.Tensor(np.random.random(self.shape)), 0
        if pb.Use_GPU==True: matrix = matrix.cuda()

        bar = tqdm(total=len(loader), ncols=pb.Tqdm_Len, desc='Getting Counterfactual Input')
        self.eval()
        with torch.no_grad():
            for batch_idx, (x, fcx, pcx, y, x_tensor, fcx_tensor, pcx_tensor, y_tensor) in enumerate(loader):
                matrix += torch.sum(x_tensor, dim=0)
                cnt += x_tensor.size()[0]
                if pb.Use_GPU == True: torch.cuda.empty_cache()
                bar.update(1)
            bar.close()
            matrix = matrix * 1.0 / cnt

        counterfactual_input = torch.unsqueeze(matrix, dim=0)
        return counterfactual_input

    def Evaluate(self, train_loader, dev_loader, test_loader, mark=''):
        if pb.Use_GPU == True: torch.cuda.empty_cache()

        fully_counterfactual_input = self.Get_Counterfactual_Input(train_loader)
        fully_counterfactual_output = self.forward(fully_counterfactual_input).cpu().data[0].numpy()

        dev_fmaf1, best_dev_cmaf1, best_x, best_y = self.Test_maF1(dev_loader, fully_counterfactual_output, rates=None)

        rates = [(best_x,best_y), (0.0,0.0), (1.0,0.0), (0.0,1.0), (0.5,0.5)]
        test_maf1s = self.Test_maF1(test_loader, fully_counterfactual_output, rates=rates)

        factual_label_fairness, counterfactual_label_fairness, factual_keyword_fairness, counterfactual_keyword_fairness = self.Test_Fairness(test_loader, fully_counterfactual_output, rate=(best_x,best_y))

        self.Save(dev_fmaf1, best_dev_cmaf1, rates, test_maf1s, factual_label_fairness, counterfactual_label_fairness, factual_keyword_fairness, counterfactual_keyword_fairness, mark=mark)

    def Test_maF1(self, test_loader, fully_counterfactual_output, rates=None):
        if pb.Use_GPU == True: torch.cuda.empty_cache()

        true_labels, factual_outputs, partial_counterfactual_outputs = [], [], []
        testbar = tqdm(total=len(test_loader), ncols=pb.Tqdm_Len, desc='Testing')
        self.eval()
        with torch.no_grad():
            for batch_idx, (x, fcx, pcx, y, x_tensor, fcx_tensor, pcx_tensor, y_tensor) in enumerate(test_loader):
                true_labels.extend(y_tensor.cpu().data.numpy())
                factual_outputs.extend(self.forward(x_tensor).cpu().data.numpy())
                partial_counterfactual_outputs.extend(self.forward(pcx_tensor).cpu().data.numpy())
                if pb.Use_GPU == True: torch.cuda.empty_cache()
                testbar.update(1)
            testbar.close()
        if pb.Use_GPU == True: torch.cuda.empty_cache()

        if rates==None:
            Dirs = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]]
            best_x, best_y, best_dev_cmaf1, cmaf1_map = 0.0, 0.0, -pb.INF, {}
            while True:
                recorded_x, recorded_y = best_x, best_y
                for i in range(len(Dirs)):
                    cur_x, cur_y, step = recorded_x, recorded_y, 0
                    while True:
                        key = '{:.2f}_{:.2f}'.format(cur_x, cur_y)
                        if key not in cmaf1_map.keys():
                            _, predict_labels = self.Counterfactual_Predict(factual_outputs, fully_counterfactual_output, partial_counterfactual_outputs, cur_x, cur_y)
                            cmaf1 = pb.Get_Report(true_labels, predict_labels)['macro_f1']
                            cmaf1_map[key] = cmaf1
                        cmaf1 = cmaf1_map[key]
                        if cmaf1 > best_dev_cmaf1:
                            best_dev_cmaf1, best_x, best_y, step = cmaf1, cur_x, cur_y, 0
                        if step>=pb.Beam_Search_Range:
                            break
                        cur_x += Dirs[i][0] * pb.Beam_Search_Step
                        cur_y += Dirs[i][1] * pb.Beam_Search_Step
                        step += 1
                if recorded_x==best_x and recorded_y==best_y:
                    break
            return cmaf1_map['{:.2f}_{:.2f}'.format(0.00,0.00)], best_dev_cmaf1, best_x, best_y
        else:
            test_maf1s = []
            for rate in rates:
                _, predict_labels = self.Counterfactual_Predict(factual_outputs, fully_counterfactual_output, partial_counterfactual_outputs, rate[0], rate[1])
                cmaf1 = pb.Get_Report(true_labels, predict_labels)['macro_f1']
                test_maf1s.append(cmaf1)
            return test_maf1s

    def Test_Fairness(self, test_loader, fully_counterfactual_output, rate):
        if pb.Use_GPU == True: torch.cuda.empty_cache()

        xs, factual_outputs, factual_labels, partial_counterfactual_outputs = [], [], [], []
        testbar = tqdm(total=len(test_loader), ncols=pb.Tqdm_Len, desc='Testing')
        self.eval()
        with torch.no_grad():
            for batch_idx, (x, fcx, pcx, y, x_tensor, fcx_tensor, pcx_tensor, y_tensor) in enumerate(test_loader):
                xs.extend(pcx)
                factual_outputs.extend(self.forward(x_tensor).cpu().data.numpy())
                partial_counterfactual_outputs.extend(self.forward(pcx_tensor).cpu().data.numpy())
                if pb.Use_GPU == True: torch.cuda.empty_cache()
                testbar.update(1)
            testbar.close()
            factual_outputs, factual_labels = self.Counterfactual_Predict(factual_outputs, fully_counterfactual_output, partial_counterfactual_outputs, 0.0, 0.0)
            counterfactual_outputs, counterfactual_labels = self.Counterfactual_Predict(factual_outputs, fully_counterfactual_output, partial_counterfactual_outputs, rate[0], rate[1])
        if pb.Use_GPU == True: torch.cuda.empty_cache()

        def Compute_Fairness(us, v):
            distance, fairness = 0.0, 0.0
            for u in us:
                distance += pb.JS(u, v)
            distance /= len(us)
            fairness = distance * 100.0
            return fairness

        uniform = [1.0/len(pb.YList) for _ in pb.YList]

        factual_label_distributions = [F.softmax(torch.Tensor(factual_output)).numpy() for factual_output in factual_outputs]
        counterfactual_label_distributions = [F.softmax(torch.Tensor(counterfactual_output)).numpy() for counterfactual_output in counterfactual_outputs]
        factual_label_fairness = Compute_Fairness(factual_label_distributions, uniform)
        counterfactual_label_fairness = Compute_Fairness(counterfactual_label_distributions, uniform)

        factual_keyword_distributions, counterfactual_keyword_distributions, w2c = [], [], {}
        for i, x in enumerate(xs):
            for word in x.split(pb.Defined_Spliter):
                if word==pb.Mask_Token:
                    continue
                if word not in w2c.keys():
                    w2c[word] = [[0 for _ in pb.YList], [0 for _ in pb.YList]]
                w2c[word][0][factual_labels[i]] += 1
                w2c[word][1][counterfactual_labels[i]] += 1
        for word in w2c.keys():
            u = w2c[word][0]
            su = sum(u)
            u = [value / su for value in u]
            v = w2c[word][1]
            sv = sum(v)
            v = [value / sv for value in v]
            factual_keyword_distributions.append(u)
            counterfactual_keyword_distributions.append(v)
        factual_keyword_fairness = Compute_Fairness(factual_keyword_distributions, uniform)
        counterfactual_keyword_fairness = Compute_Fairness(counterfactual_keyword_distributions, uniform)

        return factual_label_fairness, counterfactual_label_fairness, factual_keyword_fairness, counterfactual_keyword_fairness

    def Counterfactual_Predict(self, factual_outputs, fully_counterfactual_output, partial_counterfactual_outputs, cur_x, cur_y):
        factual_outputs = np.array(factual_outputs)
        fully_counterfactual_output = np.array(fully_counterfactual_output)
        partial_counterfactual_outputs = np.array(partial_counterfactual_outputs)
        debiased_outputs = factual_outputs - (fully_counterfactual_output * cur_x + partial_counterfactual_outputs * cur_y)
        predicted_labels = torch.max(torch.Tensor(debiased_outputs),1)[1]
        return debiased_outputs, predicted_labels.numpy()

    def Save(self, dev_fmaf1, best_dev_cmaf1, rates, test_maf1s, flf, clf, fkf, ckf, mark=''):
        id = '{}-{}-Seed={}-Start_Time={}-Current_Time={}-Epoch={}'.format(pb.Dataset_Name, self.name, pb.Seed, pb.Start_Time, pb.Get_Time(), mark)

        # torch.save(self.state_dict(), './models/'+id+'.pt')
        # pb.Pickle_Save([fdes, cdes, ftes, ctes, rates], './models/'+id+'.pickle')
        # pb.Print('Model Saved.', color='blue')

        report = '{} | '.format(id)
        report += 'dev_factual_maf1={:.2%} | '.format(dev_fmaf1)
        report += 'dev_counterfactual_maf1={:.2%} | '.format(best_dev_cmaf1)
        for i in range(len(test_maf1s)):
            report += 'test_counterfactual_maf1={:.2%}(rate={:.2f},{:.2f}) | '.format(test_maf1s[i], rates[i][0], rates[i][1])
        report += 'factual_label_fairness={} | counterfactual_label_fairness={} | '.format(flf, clf)
        report += 'factual_keyword_fairness={} | counterfactual_keyword_fairness={} | '.format(fkf, ckf)

        writer = open(pb.Save_Path, 'a+')
        writer.write(report + '\n')
        writer.close()
        print(report)

class TextCNN(SuperNetwork):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.name = 'TextCNN'
        self.sentence_max_size = pb.XMaxLen
        self.label_size = len(pb.YList)
        self.epoch = pb.Epoch
        self.lr = pb.Learning_Rate
        self.emb_dim = pb.Embedding_Dimension
        self.shape = (self.sentence_max_size, self.emb_dim)

        self.filter_num = 100
        self.kernel_list = [1, 2, 3, 4, 5]
        self.chanel_num = 1

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.chanel_num, self.filter_num, (kernel, self.emb_dim)),
            nn.ReLU(),
            nn.MaxPool2d((self.sentence_max_size - kernel + 1, 1))
        ) for kernel in self.kernel_list])

        self.dropout = nn.Dropout(pb.Dropout_Rate)

        self.fc = nn.Linear(self.filter_num * len(self.kernel_list), self.label_size)

    def forward(self, xs):
        xs = xs.unsqueeze(1)
        in_size = xs.size(0)
        out = [conv(xs) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(in_size, -1)
        out = F.dropout(out)
        out = self.fc(out)
        return out

class RoBERTa(SuperNetwork):
    def __init__(self):
        super(RoBERTa, self).__init__()
        self.name = 'RoBERTa'
        self.sentence_max_size = pb.XMaxLen
        self.label_size = len(pb.YList)
        self.epoch = pb.Epoch
        self.lr = pb.Learning_Rate
        self.emb_dim = pb.Embedding_Dimension
        self.shape = self.emb_dim

        self.hidden_layer = 100

        self.roberta = RobertaModel.from_pretrained('./roberta.base', checkpoint_file='model.pt')

        self.mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.hidden_layer),
            nn.Tanh(),
            nn.Dropout(pb.Dropout_Rate),
            nn.Linear(self.hidden_layer, self.label_size)
        )

    def forward(self, xs):
        out = self.mlp(xs)
        return out
