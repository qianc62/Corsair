# bert-serving-start -model_dir /Users/qianchen/PycharmProjects/multi_cased_L-12_H-768_A-12

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import sys
from gensim.models import KeyedVectors
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
# from fairseq.models.roberta import RobertaModel






if __name__ == "__main__":

    # print(stats.entropy([0.33,0.33,0.33], [0.33,0.33,0.33]))
    # print(stats.entropy([0.4,0.2,0.7], [0.33,0.33,0.33]))

    train_examples, dev_examples, test_examples = Read_Data()

    seeds = Generate_Seeds(train_examples)
