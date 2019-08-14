import numpy as np
from collections import defaultdict, OrderedDict
import re
import warnings
import sys
import time
import _pickle as cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import autograd

       
def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        sent.append(rev["y"])
        if rev["split"]==cv:            
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]     
  

class CNN_Text(nn.Module):
    
    def __init__(self, D, C, Ci, Co, Ks):
        super(CNN_Text, self).__init__()
        # C is number of classes.       
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        dropout=0.5
        self.dropout = nn.Dropout(dropout)
        #self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.fc1 = nn.Linear(len(Ks)*Co, 100)
        self.fc2 = nn.Linear(100, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # x already is a 4-d tensor with word2vec embeddings. 
        #x = self.embed(x)  # (N, W, D)
        #if self.args.static:
        #    x = Variable(x)
        # x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        if GUTSOPEN:
           # dump the outout in 71st filter (counting from index 1) pos sent
           # x[0] is first conv filter's output  
           # dump the outout in 6th filter (counting from index 1) neg sent. see file convolutionActivations.txt for details.
           #ofile3.write(str(x[0][:,70,:].data))
           ofile3.write(str(x[0][:,5,:].data))
           ofile3.write('\n')



        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        
        if GUTSOPEN:  
           for mp in x:
              ofile3.write(str(mp.data))
              ofile3.write('\n')

        x = torch.cat(x, 1)
        

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        # Remove dropout for now
        #x = self.dropout(x)  # (N, len(Ks)*Co)
        #logit =self.fc1(x)
        x = F.relu(self.fc1(x))
        logit = self.fc2(x)  # (N, C)
        return logit
   
if __name__=="__main__":
    print("loading data...")
    x = cPickle.load(open("mr.p","rb"))
    CNNModelFile = 'CNNPyTorchModel'
    #GUTSOPEN = False 
    GUTSOPEN = True 
    revs, W, W2, word_idx_map,idx_word_map, vocab = x[0], x[1], x[2], x[3], x[4],x[5]
    print("data loaded!")
    batch_size = 20
    img_w = 300 # 300 dimensional Word2vec embeddings.
    n_epochs = 30
    D = img_w
    C = 2 # number of classes, 
    Ci = 1
    Co = 200
    Ks = [3,4,5]
    #Ks = [3]
    lrnR = 0.01
    U = W
    print("U.shape="+str(U.shape))
    results = []
    np.random.seed(123)
    r = range(0,1)    
    for i in r:
        # datasets[0] has train set and datasets[1] has test. 
	# datasets' are  matrices of ids (N x max_l) where the row 'i' correponds to the sequence of words ids (in the matrix U) in the sentence 'i'. 
        datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=56,k=300, filter_h=5)
        img_h = len(datasets[0][0])-1 # This is equal to number of words in each image after padding i.e max_l
        print(" img_h: "+str(img_h))
        # Let's check the word2vec vectors.
        # dump out the first 4 revs and their word2vec. Some of them may not be in the train set i.e datasets[0] if their split==0.  
        '''
        trainCnt=0
        for l in range(4):
           if revs[l]['split'] != 0:
              # part of trainset.
              print("revs["+str(l)+']["text"]'+str(revs[l]['text']))
              print('revs['+str(l)+']["split"]'+str(revs[l]['split']))
              print('label='+ str( datasets[0][trainCnt][-1] ))
              for k in range(img_h):
                 print(str(k)+" word="+idx_word_map[ datasets[0][trainCnt][k]]+' wordId='+str(datasets[0][trainCnt][k])+str(U[ datasets[0][trainCnt][k] ]))
              trainCnt +=1
        sys.exit() 
        '''
        # dump put the test-set sentences. 
        ofile1 = open('testSet.txt','w')
        ofile3 = open('intermediateTensors.txt','w')
        testCnt = 0 
        for l in range(len(revs)):
           if revs[l]['split'] == 0:   
              ofile1.write(str(testCnt)+' Target='+str(revs[l]['y'] )+' '+str(revs[l]['text']))
              testCnt +=1
              ofile1.write('\n')
        #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
        #extra data (at random)
        # datatsets[0].shape[0] == number of total training sentences and and datasets[0][i] is the ith sentence's sequence of words ids in wor2vec master matrix U.  
        # train set words indices and label are in datasets[0]
        # datasets[0] and datasets[1] are 2-d numpy arrays, converted from list(list). 
        # Now shuffle the entire matrix new_data which has multiple rows of batch_size. 
        new_data = datasets[0]
        new_data = np.random.permutation(new_data)
        n_batches = new_data.shape[0]/batch_size
        n_train_batches = int(np.round(n_batches*0.9))
        #divide train set into train/val sets 
        train_set = new_data[:n_train_batches*batch_size,:]
        val_set = new_data[n_train_batches*batch_size:,:]
        train_set_x = train_set[:,:img_h]
        train_set_y = train_set[:,-1]
        val_set_x = val_set[:,:img_h]
        val_set_y = val_set[:,-1]
        n_val_batches = n_batches - n_train_batches    
        # test set words indices and label are in datasets[1]
        # here pass on the test-sentence index that you want to analyze
        if GUTSOPEN:
           #index = 8 # sent # 9 in testset
           index = 1045 # sent # 1046 in testset
           # here we used (index-1):index to select the index sent to retain test_set_x as 2d numpy array of size 1x64. If we woud have used only index to select we would have gotten a 1-d numpy of size 64.
           test_set_x = datasets[1][(index):(index+1),:img_h]
           test_set_y = np.asarray(datasets[1][(index):(index+1),-1],"int32")
        else:
           test_set_x = datasets[1][:,:img_h]
           test_set_y = np.asarray(datasets[1][:,-1],"int32")


        # instantiate the cnn model from CNN_Text class and criterion from nn.CrossEntropyLoss() class
        cnn = CNN_Text(D, C, Ci, Co, Ks)
        criterion = nn.CrossEntropyLoss()
        # now load the CNN model
        cnn.load_state_dict(torch.load(CNNModelFile))

        #optimizer = torch.optim.SGD(cnn.parameters(), lr=lrnR)
        xTest2d = U[test_set_x.reshape(-1)]
        print(' test_set_x.shape='+str(test_set_x.shape))
        xTest4d = xTest2d.reshape( test_set_x.shape[0],1,img_h,-1)
        xTestTensor = torch.from_numpy(xTest4d).float()
        yTestTensor = torch.from_numpy(test_set_y).long()
        logitTest = cnn(xTestTensor)
        loss = criterion(logitTest,yTestTensor)
        corrects = (torch.max(logitTest, 1)[1].view(yTestTensor.size()).data == yTestTensor.data).sum()
        accuracy = 100.0 * float(corrects)/test_set_x.shape[0]
        print(" Test accuracy="+str(accuracy) )
        print(' Test loss='+str(loss))
        print('corrects='+str( corrects) + "total="+str( yTestTensor.size() ))

        # convert to posteriors by applying softmax to the activations in logitTest
        ofile2 =open('CNNProbs.txt','w')
        softMaxFunc = torch.nn.Softmax(1) # apply soift max across the second dimension. The first is the batch size. 
        probsTensor = softMaxFunc(logitTest)
        for l in range(test_set_x.shape[0]):
           ofile2.write("Target="+str( yTestTensor.data[l])+" "+str(probsTensor.data[l]))
           ofile2.write('\n')


