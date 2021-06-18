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



class Example:
    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.fully_counterfactual_text = []
        self.partial_counterfactual_text = []

class MyAllDataset():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.train_examples = []
        self.dev_examples = []
        self.test_examples = []

        self.Read_Data()

    def Read_Data(self):
        # output dataset's name
        print('Dataset:{}'.format(self.dataset_name))
        pb.Print_Line(color='blue')

        if pb.EDA==False:
            train_datapath = './data/' + self.dataset_name + '.train.jsonl'
        else:
            pb.Print('pb.EDA=Ture', color='blue')
            train_datapath = './data/data.eda/' + self.dataset_name + '.train.eda.jsonl'
        dev_datapath = './data/' + self.dataset_name + '.dev.jsonl'
        test_datapath = './data/' + self.dataset_name + '.test.jsonl'

        # read data
        def Read_from_Datapath(data_path):
            examples = []
            for line in open(data_path).read().split('\n'):
                if '{' in line:
                    linemap = json.loads(line.lower())
                    if len(linemap['text'].strip())>0 and len(linemap['label'].strip())>0:
                        examples.append( Example(linemap['text'], linemap['label']) )
            return examples
        train_examples = Read_from_Datapath(train_datapath)
        dev_examples   = Read_from_Datapath(dev_datapath)
        test_examples  = Read_from_Datapath(test_datapath)

        pb.YList = sorted(list(set([example.label for example in train_examples+dev_examples+test_examples])))

        if pb.EDA==True:
            train_examples = self.Label_Balance(train_examples)

        # Conform
        def Conform_Dev_Test(dev_examples, test_examples):
            examples = dev_examples + test_examples
            label2examples = {}
            for example in examples:
                label = example.label
                if label not in label2examples:
                    label2examples[label] = []
                label2examples[label].append(example)
            dev_examples_, test_examples_ = [], []
            for key in label2examples.keys():
                subexamples = label2examples[key]
                random.shuffle(subexamples)
                seperator = int(len(subexamples) / 2)
                dev_examples_.extend(subexamples[:seperator])
                test_examples_.extend(subexamples[seperator:])
            pb.Print('Dev and Test Conformed.', color='green')
            return dev_examples_, test_examples_
        dev_examples, test_examples = Conform_Dev_Test(dev_examples, test_examples)

        # analysis
        random.shuffle(train_examples)
        random.shuffle(dev_examples)
        random.shuffle(test_examples)
        trLen, deLen, teLen = len(train_examples), len(dev_examples), len(test_examples)
        train_examples = train_examples[:min(len(train_examples), pb.Train_Example_Num_Control)]
        dev_examples = dev_examples[:min(len(dev_examples), int(len(train_examples)*1.0/trLen*deLen))]
        test_examples = test_examples[:min(len(test_examples), int(len(train_examples)*1.0/trLen*teLen))]
        trLen, deLen, teLen = len(train_examples), len(dev_examples), len(test_examples)
        alLen = trLen + deLen + teLen
        print('#train_examples: {}({:.2%})'.format(trLen, trLen * 1.0 / alLen))
        print('#dev_examples: {}({:.2%})'.format(deLen, deLen * 1.0 / alLen))
        print('#test_examples: {}({:.2%})'.format(teLen, teLen * 1.0 / alLen))
        pb.Print_Line(color='blue')

        # initialize
        def Init_Public(train_examples, dev_examples, test_examples):
            examples = train_examples + dev_examples + test_examples
            bar = tqdm(total=len(examples), ncols=pb.Tqdm_Len)
            for i, example in enumerate(examples):
                # print(example.text)
                sentence = example.text
                # print(sentence)

                if pb.Base_Model=='TextCNN':
                    example.text = pb.WordSegmentation(example.text)
                else:
                    example.text = example.text.split(' ')
                example.text = [word.strip() for word in example.text if len(word.strip())>0]

                keywords = jieba.analyse.extract_tags(sentence, topK=pb.INF, withWeight=True)
                keywords_map = {}
                for item in keywords:
                    keywords_map[item[0]] = item[1]
                for j in range(len(example.text)):
                    example.fully_counterfactual_text.append(pb.Mask_Token)
                    if example.text[j] in keywords_map:
                        example.partial_counterfactual_text.append(pb.Mask_Token)
                    else:
                        example.partial_counterfactual_text.append(example.text[j])

                bar.set_description( 'Word Segmentating and Partial_Counterfactual Processing')
                bar.update(1)
            bar.close()
            for x in [example.text for example in examples]:
                pb.XMaxLen = min(max(pb.XMaxLen, len(x)), pb.XMaxLenLimit)
        Init_Public(train_examples, dev_examples, test_examples)
        print('pb.XMaxLen={}'.format(pb.XMaxLen))
        print('pb.YList={} {}'.format(len(pb.YList), pb.YList))
        pb.Print_Line(color='blue')

        # probability distributions
        train_distribution = pb.Train_Distribution = [0 for _ in range(len(pb.YList))]
        dev_distribution = [0 for _ in range(len(pb.YList))]
        test_distribution = [0 for _ in range(len(pb.YList))]
        for e in train_examples: train_distribution[pb.YList.index(e.label)] += 1
        for e in dev_examples:   dev_distribution[pb.YList.index(e.label)] += 1
        for e in test_examples:  test_distribution[pb.YList.index(e.label)] += 1
        train_distribution = [x * 1.0 / sum(train_distribution) for x in train_distribution]
        dev_distribution = [x * 1.0 / sum(dev_distribution) for x in dev_distribution]
        test_distribution = [x * 1.0 / sum(test_distribution) for x in test_distribution]
        print('train_distribution: [', end='')
        for v in train_distribution: print('{:.2%}'.format(v), end=' ')
        print('] {}'.format('Balanced' if pb.EDA==True else 'Raw'))
        print('dev_distribution:   [', end='')
        for v in dev_distribution: print('{:.2%}'.format(v), end=' ')
        print(']')
        print('test_distribution:  [', end='')
        for v in test_distribution: print('{:.2%}'.format(v), end=' ')
        print(']')
        pb.Print_Line(color='blue')

        # MASK ratio
        examples = train_examples + dev_examples + test_examples
        Ratio = 0.0
        for example in examples:
            up = len([word for word in example.partial_counterfactual_text if word == pb.Mask_Token])
            down = len(example.text)
            Ratio += up * 1.0 / down
        Ratio = Ratio * 1.0 / len(examples)
        print('{:.2%} MASKed ({:.2%} is context)'.format(Ratio, 1.0 - Ratio))

        self.train_examples = train_examples
        self.dev_examples   = dev_examples
        self.test_examples  = test_examples

    def Label_Balance(self, examples):

        examples_list = [[] for _ in pb.YList]
        sampled_examples_list = [[] for _ in pb.YList]

        for example in examples:
            index = pb.YList.index(example.label)
            examples_list[index].append(example)
            sampled_examples_list[index].append(example)
        sample_num = int(np.max([len(obj) for obj in examples_list]) * 1.0)

        balanced_examples = []
        for i in range(len(sampled_examples_list)):
            while (len(sampled_examples_list[i]) < sample_num):
                example = random.choice(examples_list[i])
                sampled_examples_list[i].append( copy.deepcopy(example) )
            balanced_examples.extend(sampled_examples_list[i])
        return balanced_examples

class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        index %= self.__len__()

        x = self.examples[index].text
        fcx = self.examples[index].fully_counterfactual_text
        pcx = self.examples[index].partial_counterfactual_text
        x = pb.Defined_Spliter.join(x)
        fcx = pb.Defined_Spliter.join(fcx)
        pcx = pb.Defined_Spliter.join(pcx)

        y = self.examples[index].label
        x_tensor = self.Generate_X_Tensor(x)
        fcx_tensor = torch.Tensor(np.array([0.0]))
        pcx_tensor = self.Generate_X_Tensor(pcx)
        y_tensor = self.Generate_Y_Tensor(y)
        return x, fcx, pcx, y, x_tensor, fcx_tensor, pcx_tensor, y_tensor

    def Generate_X_Tensor(self, text):
        pass

    def Generate_Y_Tensor(self, label):
        tensor = torch.zeros(len(pb.YList))
        tensor[pb.YList.index(label)] = 1
        tensor = torch.argmax(tensor)
        if pb.Use_GPU == True:
            tensor = tensor.cuda()
        return tensor

class MyDataset_TextCNN(MyDataset):
    def __init__(self, embedding, word2id, examples):
        super(MyDataset, self).__init__()
        self.embedding = embedding
        self.word2id = word2id
        self.examples = examples
        self.sentence_max_size = pb.XMaxLen
        self.embedding_dim = pb.Embedding_Dimension

    def Generate_X_Tensor(self, text):
        words = text
        tensor = torch.zeros([self.sentence_max_size, self.embedding_dim])
        for index in range(0, self.sentence_max_size):
            if index >= len(words):
                break
            else:
                word = words[index]
                if word in self.word2id:
                    vector = self.embedding.weight[self.word2id[word]]
                    tensor[index] = vector
                elif word.lower() in self.word2id:
                    vector = self.embedding.weight[self.word2id[word.lower()]]
                    tensor[index] = vector
        if pb.Use_GPU == True:
            tensor = tensor.cuda()
        return tensor

class MyDataset_RoBERTa(MyDataset):
    def __init__(self, examples, roberta):
        super(MyDataset, self).__init__()
        self.roberta = roberta
        self.examples = examples

    def Generate_X_Tensor(self, text):
        text = ' '.join(text)
        tokens = self.roberta.encode(text)
        tokens = tokens[:min(512, len(tokens))]
        last_layer_feature = self.roberta.extract_features(tokens)[-1][0]
        if pb.Use_GPU == True:
            last_layer_feature = last_layer_feature.cuda()
        return last_layer_feature
