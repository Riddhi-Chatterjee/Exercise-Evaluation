import datasetHandler
import LSTM
import signal
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from os.path import exists
import math
import shutil

def signal_handler(sig, frame):
    print('\nExiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#Settings:
numClasses = 1
seqLen = 20
samplingPeriod = 0.2
model = LSTM.LSTM(numClasses, seqLen)

ch = input("Use existing datasets? Y/N: ")
if ch=="N":
    dh = datasetHandler.datasetHandler(seqLen, samplingPeriod)
    
    dh.createTestingDataset()
    
    dh.shuffleDataset("datasets/Test_Datasets", "master_dataset.txt")
    dh.splitDataset("datasets/Test_Datasets", "master_dataset.txt")
    
FILE = "checkpoint.pth"
checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])
epoch = checkpoint['epoch']
datasetNum = checkpoint['dataset_number']
batchNum = checkpoint['batch_number']

print("\nTrained upto (of training dataset):")
print("epoch = "+str(epoch))
print("datasetNum = "+str(datasetNum))
print("batchNum = "+str(batchNum)+"\n")

datasetNum = 1
hits = 0
misses = 0
while(exists("datasets/Test_Datasets/dataset_"+str(datasetNum)+".txt")):
    dataset = datasetHandler.LSTMdataset("datasets/Test_Datasets", datasetNum)
    train_loader = DataLoader(dataset=dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=0)
    
    #total_samples = len(dataset)
    #n_iterations = math.ceil(total_samples/model.batch_size)
    ##########################################################
    
    for i, (inputs, labels) in enumerate(train_loader):
        y_pred = model(inputs)
        y_pred = y_pred.view(y_pred.size(0))
        
        labels = labels.view(labels.size(0))
        labels = labels.long()
        
        if (y_pred.item() >= labels.item() - 0.01) and (y_pred.item() <= labels.item() + 0.01):
            hits+=1
        else:
            misses+=1
        
        print("Hits : "+str(hits)+"  Misses : "+str(misses), end="\r")
        
    ##########################################################
    
    datasetNum += 1

print("y_pred:"+str(y_pred.item())+"  labels:"+str(labels.item()))
print("Hits : "+str(hits)+"  Misses : "+str(misses))
accuracy = (hits/(hits+misses))*100
print("Accuracy = "+str(accuracy)+"%")

signal_handler(0, 0)