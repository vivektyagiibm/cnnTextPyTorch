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
        # total number of words cannot be more than max_l+2*pad
        if len(x) == max_l+2*pad:
           break
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
        # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        # Remove dropout for now
        #x = self.dropout(x)  # (N, len(Ks)*Co)
        #logit =self.fc1(x)
        x = F.relu(self.fc1(x))
        logit = self.fc2(x)  # (N, C)
        return logit
   
if __name__=="__main__":
    print("loading data...")
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map,idx_word_map, vocab = x[0], x[1], x[2], x[3], x[4],x[5]
    print("data loaded!")
    batch_size = 20
    img_w = 300 # 300 dimensional Word2vec embeddings.
    n_epochs = 10 # run for a  fixed nunmebr of epochs
    D = img_w
    C = 2 # number of classes, 
    Ci = 1 # input channel
    Co = 200 # number of filter maps
    Ks = [3,4,5] # three sizes of filters, 3xD, 4xD, 5xD, and D is the embedding dimention
    #Ks = [3]
    lrnR = 0.01 # learning rate to use in SGD
    U = W
    print("U.shape="+str(U.shape))
    results = []
    np.random.seed(123)
    r = range(0,1) # run just one CV split.   
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
        test_set_x = datasets[1][:,:img_h]
        test_set_y = np.asarray(datasets[1][:,-1],"int32")


        # instantiate the cnn model from CNN_Text class and criterion from nn.CrossEntropyLoss() class
        cnn = CNN_Text(D, C, Ci, Co, Ks)
        criterion = nn.CrossEntropyLoss()
        #optimizer = torch.optim.SGD(cnn.parameters(), lr=lrnR)
        epoch =0 
        steps = 0
        while (epoch < n_epochs):
           start_time = time.time()
           epoch = epoch + 1
           for index in np.random.permutation(range(n_train_batches)):
           #for index in (range(1)):
              x = train_set_x[index*batch_size:(index+1)*batch_size]
              y = train_set_y[index*batch_size:(index+1)*batch_size]
              #print("x.shape="+str(x.shape))
              #print("y.shape="+str(y.shape))
              x2dBatch = U[x.reshape(-1)] 
              #print(" x2dBatch.shape=  "+ str(x2dBatch.shape))
              # now reshape it to 4-d np array (batch_size,1,img_h, img_w)
              # due to C style reshaping (raveling, unraveling) of numpy array b y default, the following rehape is fine and the first index will indeed corresponds to the batch_size and the last index to the 300 dimensional word2vecs of words. 
              x4dBatch = x2dBatch.reshape(batch_size,1,img_h,-1 )
              #print(" After reshaping to 4d x4dBatch.shape=  "+ str(x4dBatch.shape))
              # now convert these to pyTorch Tensor
              xTensor = torch.from_numpy(x4dBatch).float()
              yTensor = torch.from_numpy(y).long()

              #optimizer.zero_grad()
              cnn.zero_grad()
              logit = cnn(xTensor)
              #print('logit='+str(logit))
              #print('yTensor='+str(yTensor))
              #loss = F.cross_entropy(logit,yTensor)
              loss = criterion(logit,yTensor)
              loss.backward()
             
              
              for p in cnn.parameters():
                 p.data.add_(-lrnR, p.grad.data)
              
              '''
              for p in cnn.parameters():
                 print('p.data='+str(p.data))
              '''
              # here dump out the parameter.
              '''
              for p in cnn.parameters():
                 # in place update of the parameters
                 print('Before update p.data='+str(p.data))
                 print('p.grad.data='+str(p.grad.data))
                 p.data.add_(-lrnR, p.grad.data)
                 print('After update p.data='+str(p.data))
              '''
              #optimizer.step()
              #print('index='+str(index)+' loss='+str(loss))
              steps += 1
              if  steps % 20 == 0:
                 corrects = (torch.max(logit, 1)[1].view(yTensor.size()).data == yTensor.data).sum()
                 accuracy = 100.0 * corrects/batch_size
                 print("Epoch="+str(epoch)+"steps="+str(steps)+ " accuracy="+str(accuracy) )
                 print('index='+str(index)+' loss='+str(loss))
                 # validation set
                 xVal2d = U[val_set_x.reshape(-1)]
                 xVal4d = xVal2d.reshape( val_set_x.shape[0],1,img_h,-1)
                 xValTensor = torch.from_numpy(xVal4d).float()
                 # yValTensor is a 1-d tensor of targets as long ints
                 yValTensor = torch.from_numpy(val_set_y).long() 
                 logitVal = cnn(xValTensor)
                 loss = criterion(logitVal, yValTensor)
                 corrects = (torch.max(logitVal,1)[1].view(yValTensor.size()).data == yValTensor.data).sum()
                 accuracy = 100.0 * float(corrects)/val_set_x.shape[0]
                 print(" Val accuracy="+str(accuracy) )
                 print(' Val loss='+str(loss))
                 print('Val corrects='+str( corrects) + "total="+str( yValTensor.size() ))
                 # Now also evaluate the test_set
                 xTest2d = U[test_set_x.reshape(-1)]
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
    # save the trained model.
    torch.save(cnn.state_dict(),'CNNPyTorchModel')

