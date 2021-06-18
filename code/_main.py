from MyData import *
from MyModel import *



def MAIN():

    # initialize
    pb.Start_Time = pb.Get_Time()
    pb.Seed = random.randint(0, pb.INF)
    pb.XMaxLen = 0
    pb.YList = []
    pb.Use_GPU = torch.cuda.is_available()

    # seed setting
    random.seed(pb.Seed)
    os.environ['PYTHONHASHSEED'] = str(pb.Seed)
    np.random.seed(pb.Seed)
    torch.manual_seed(pb.Seed)
    torch.cuda.manual_seed(pb.Seed)
    torch.cuda.manual_seed_all(pb.Seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # output configuration
    pb.Print('pb.Dataset_Name={}'.format(pb.Dataset_Name), color='blue')
    pb.Print('pb.Base_Model={}'.format(pb.Base_Model), color='blue')
    pb.Print('pb.EDA={}'.format(pb.EDA), color='blue')
    pb.Print('pb.Weight={}'.format(pb.Weight), color='blue')
    pb.Print('pb.Save_Path={}'.format(pb.Save_Path), color='blue')
    pb.Print('pb.Use_GPU={}'.format(pb.Use_GPU), color='blue')
    pb.Print_Line(color='blue')

    # prepare a specific dataset
    myAllDataset = MyAllDataset(pb.Dataset_Name)

    if pb.Base_Model=='TextCNN':
        if pb.Dataset_Name not in ['Suning', 'Taobao']:
            # English dictionary
            w2v_path = './w2v/glove.300d.en.txt'
        else:
            # Chinese dictionary
            w2v_path = './w2v/fasttext.300d.zh.txt'
        print(w2v_path)
        w2v_pickle = w2v_path + '.pickle'
        if os.path.exists(w2v_pickle)==False:
            wv, word2id = KeyedVectors.load_word2vec_format(w2v_path, binary=False), {}
            for i, word in enumerate(wv.index2word): word2id[word] = i
            embedding = nn.Embedding.from_pretrained(torch.FloatTensor(wv.vectors))
            pb.Pickle_Save([embedding, word2id], w2v_pickle)
        else:
            [embedding, word2id] = pb.Pickle_Read(w2v_pickle)
        embedding.weight.requires_grad = False

        network = TextCNN()
        if pb.Use_GPU == True:
            network = network.cuda()

        train_dataset = MyDataset_TextCNN(embedding, word2id, myAllDataset.train_examples)
        dev_dataset   = MyDataset_TextCNN(embedding, word2id, myAllDataset.dev_examples)
        test_dataset  = MyDataset_TextCNN(embedding, word2id, myAllDataset.test_examples)

    elif pb.Base_Model=='RoBERTa':
        network = RoBERTa()
        if pb.Use_GPU == True:
            network = network.cuda()

        train_dataset = MyDataset_RoBERTa(myAllDataset.train_examples, network.roberta)
        dev_dataset = MyDataset_RoBERTa(myAllDataset.dev_examples, network.roberta)
        test_dataset = MyDataset_RoBERTa(myAllDataset.test_examples, network.roberta)

    train_loader = DataLoader(train_dataset, batch_size=pb.Train_Batch_Size, shuffle=pb.DataLoader_Shuffle)
    dev_loader = DataLoader(dev_dataset, batch_size=pb.DevTest_Batch_Size, shuffle=pb.DataLoader_Shuffle)
    test_loader = DataLoader(test_dataset, batch_size=pb.DevTest_Batch_Size, shuffle=pb.DataLoader_Shuffle)

    if pb.Operation == 'Train':
        network.Train(train_loader, dev_loader, test_loader)

if __name__ == "__main__":
    # read configuration
    print('sys.argv={}'.format(sys.argv))
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--Dataset_Name' and i+1<len(sys.argv):
            pb.Dataset_Names = [sys.argv[i+1]]
        if sys.argv[i] == '--Base_Model' and i+1<len(sys.argv):
            pb.Base_Model = sys.argv[i+1]
        if sys.argv[i] == '--EDA' and i+1<len(sys.argv):
            pb.EDA = True if sys.argv[i+1]=='True' else False
        if sys.argv[i] == '--Weight' and i+1<len(sys.argv):
            pb.Weight = True if sys.argv[i+1]=='True' else False
        if sys.argv[i] == '--Save_Path' and i+1<len(sys.argv):
            pb.Save_Path = sys.argv[i+1]

    if pb.Operation == 'Test':
        pb.Dataset_Name = pb.Evaluate_Model.split('./models/')[-1].split('-')[0]
        pb.Operation_Times = 1

    if pb.Base_Model == 'RoBERTa':
        pb.Embedding_Dimension = 768

    if pb.Dataset_Name in ['Taobao', 'Suning']:
        pb.EDA = False

    if pb.EDA == True:
        pb.Epoch = 20

    for _ in range(pb.Operation_Times):
        for name in pb.Dataset_Names:
            pb.Dataset_Name = name
            MAIN()
