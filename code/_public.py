from bert_serving.client import BertClient
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from progressbar import *
from scipy import stats
from sklearn.manifold import TSNE
from time import sleep
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import _public as pb
import copy
import jieba
import jieba.analyse
import json
import logging
import math
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pickle
import random
import scipy.stats
import scipy.stats as stats
import sklearn.metrics as metrics
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# All Hyperparameters

# variables
Dataset_Names = ['Twitter']
# ['HyperPartisan','Twitter','ARC','SCIERC','ChemProt','Economy','20News','Parties','Yelp_Hotel','Taobao','Suning']
EDA = False
Weight = False
Base_Model = 'TextCNN'
# TextCNN RoBERTa
Save_Path = './Results.txt'

# public
INF = 999999999
Epsilon = 1e-6
Lowercases = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
Start_Time = ''
Seed = 0
Dataset_Name = ''
XMaxLen = 0
XMaxLenLimit = 300
YList = []
Mask_Token = '[MASK]'
Train_Example_Num_Control = 80000
Use_GPU = False
Tqdm_Len = 80
Train_Distribution = []
# training
Operation = 'Train'
Operation_Times = 1
Counterfactual_Input_Control = 'Average'
DataLoader_Shuffle = False
Train_Batch_Size, DevTest_Batch_Size = 8, 8
Epoch = 20
Learning_Rate = 2e-5
Embedding_Dimension = 300
Dropout_Rate = 0.1
Defined_Spliter = '____'
# testing
Beam_Search_Step = 0.1
Beam_Search_Range = 20
Evaluate_Model = './models/.pt'



# public functions
def Print(string, color=''):
    if color=='red':
        print('\033[31m' + str(string) + '\033[0m')
    elif color == 'green':
        print('\033[32m' + str(string) + '\033[0m')
    elif color=='blue':
        print('\033[34m' + str(string) + '\033[0m')
    else:
        print(str(string))

def Print_Line(color=''):
    string = '------------------------------------------------------------'
    Print(string, color=color)

def Get_Time():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())

def Pickle_Save(variable, path):
    with open(path, 'wb') as file:
        pickle.dump(variable, file)
    Print("Pickle Saved {}".format(path), color='blue')

def Pickle_Read(filepath):
    with open(filepath, 'rb') as file:
        obj = pickle.load(file)
    Print("Pickle Read", color='blue')
    return obj

def Max_Index(array):
    max_index = 0
    for i in range(len(array)):
        if(array[i]>array[max_index]):
            max_index = i
    return max_index

# def Min_Index(array):
#     min_index = 0
#     for i in range(len(array)):
#         if(array[i]<array[min_index]):
#             min_index = i
#     return min_index

def Get_Report(true_labels, pred_labels):
    true_labels = [int(v) for v in true_labels]
    pred_labels = [int(v) for v in pred_labels]
    # label_list = sorted(list(set(true_labels+pred_labels)))
    # label_list = YList
    macro_recall = metrics.recall_score(true_labels, pred_labels, average='macro')
    # macro_recall = -1.0
    micro_recall = metrics.recall_score(true_labels, pred_labels, average='micro')
    # micro_recall = -1.0
    macro_precision = metrics.precision_score(true_labels, pred_labels, average='macro')
    # macro_precision = -1.0
    micro_precision = metrics.precision_score(true_labels, pred_labels, average='micro')
    # micro_precision = -1.0
    # macro_f1 = metrics.f1_score(true_labels, pred_labels, average='macro')
    # macro_f1 = metrics.f1_score(y_true=true_labels, y_pred=pred_labels, average='macro')
    micro_f1 = metrics.f1_score(true_labels, pred_labels, average='micro')
    # micro_f1 = -1.0
    acc = metrics.accuracy_score(true_labels, pred_labels)
    # acc = -1.0
    auc = metrics.roc_auc_score(true_labels, pred_labels) if len(label_list)==2 else -0.0
    # auc = -1.0
    # confusion_matrix = metrics.confusion_matrix(true_labels, pred_labels, labels=label_list)  if len(label_list)<=20 else [[]]
    confusion_matrix = [[]]

    report_map = {}
    report_map['macro_recall'] = macro_recall
    report_map['micro_recall'] = micro_recall
    report_map['macro_precision'] = macro_precision
    report_map['micro_precision'] = micro_precision
    report_map['macro_f1'] = macro_f1
    report_map['micro_f1'] = micro_f1
    report_map['acc'] = acc
    report_map['auc'] = auc
    report_map['confusion_matrix'] = confusion_matrix
    return report_map

def Plot_Line(line, legend=None, index=None, title='', xticks=None):
    x = range(len((line)))
    label = str(legend) if legend!=None else ''
    plt.plot(x, line, color='r', linewidth=1.0, marker='.', markersize=4, linestyle='-', label=label)
    plt.legend()
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('./figs/Plot_Lines_'+Get_Time())
    plt.show()

def Plot_Lines(lines, legends=None, index=None, title='', xticks=None):
    colors = ['blue', 'blue', 'red', 'red', 'magenta', 'cyan', 'brown', 'black', 'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']
    markers = ['.', '+', '.', '+', '+', '*', '.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', 'h', 'H',  'x', 'D', 'd', '|','_']
    linestyles = ['-', '--', '-', '--']

    x = range(len((lines[0])))
    for i in range(len(lines)):
        label = str(legends[i]) if legends!=None else ''
        plt.plot(x, lines[i], color=colors[i], linewidth=1.0, marker=markers[i], markersize=4, linestyle=linestyles[i], label=label)
    if index!=None: plt.axvline(index)
    plt.legend()
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    if xticks!=None: plt.xticks(range(len(lines[0])), xticks, fontsize=2, rotation=0)
    plt.savefig('./figs/Plot_Lines_'+Get_Time())
    plt.show()

def Plot_tSNE(matrix, labels, title=''):
    colors = ['blue', 'red', 'magenta', 'cyan', 'brown', 'black', 'aliceblue', 'antiquewhite', 'aqua','aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown','burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson','cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta','darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue','darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue','firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod','gray', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki','lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan','lightgoldenrodyellow', 'lightgreen', 'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen','lightskyblue', 'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen','magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen','mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream','mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered','orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff','peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon','sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow','springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white','whitesmoke', 'yellow', 'yellowgreen']
    # tsne = TSNE(n_components=2)
    tsne = TSNE(n_components=2, init='pca')
    tsne.fit_transform(matrix)
    compressed_matrix = tsne.embedding_
    x = [array[0] for array in compressed_matrix]
    y = [array[1] for array in compressed_matrix]
    c = [colors[label%len(colors)] for label in labels]
    plt.scatter(x, y, c=c)
    plt.title(title)
    plt.savefig('./figs/Plot_tSNE_' + Get_Time())
    plt.show()

# def Argmax_K(array, K):
#     minV = min(array) - 1.0
#     array = np.array(array)
#     indexs = []
#     for i in range(K):
#         index = np.argmax(array)
#         indexs.append(index)
#         array[index] = minV
#     return indexs

def WordSegmentation(sentence):
    words = list(jieba.cut(sentence))
    return words

def KL(x, y):
    return scipy.stats.entropy(x, y)

def JS(x, y):
    x = np.array(x)
    y = np.array(y)
    z = (x+y)/2.0
    js = 0.5*KL(x,z)+0.5*KL(y,z)
    return js

def Map_To_Sorted_List(map):
    x, y = [], []
    for item in sorted(map.items(), key=lambda item:item[1]):
        x.append(item[0])
        y.append(item[1])
    return x, y

def TTest_P_Value(array1, array2):
    sta, pv = stats.ttest_ind(array1, array2, equal_var=True)
    return pv
