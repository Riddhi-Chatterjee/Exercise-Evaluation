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

def signal_handler(sig, frame):
    checkpoint = {
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict(),
    "dataset_number" : datasetNum,
    "batch_number" : batchNum
    }
    FILE = "checkpoint.pth"
    torch.save(checkpoint, FILE)
    
    print('\nExiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

#Settings:
numClasses = 1
seqLen = 20
samplingPeriod = 0.2
learning_rate = 0.01
num_epochs = 2001
model = LSTM.LSTM(numClasses, seqLen)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
printingBatch = 0
printingDataset = 1

datasetNum = 1
epoch = 0
batchNum = 0
loss = "Dummy Initialisation"

ch = input("Use existing datasets? Y/N: ")
if ch=="N":
    dh = datasetHandler.datasetHandler(seqLen, samplingPeriod)
    
    dh.createTrainingDataset()
    
    dh.shuffleDataset("datasets/Train_Datasets", "master_dataset.txt")
    dh.splitDataset("datasets/Train_Datasets", "master_dataset.txt")
    
    with open("checkpoint.pth", "w") as c:
        pass
else:
    ch1 = input("Start training from scratch? Y/N: ")
    if ch1 == "Y":
        with open("checkpoint.pth", "w") as c:
            pass
    else:
        FILE = "checkpoint.pth"
        checkpoint = torch.load(FILE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        epoch = checkpoint['epoch']
        datasetNum = checkpoint['dataset_number']
        batchNum = checkpoint['batch_number']
        with open("checkpoint.pth", "w") as c:
            pass
 
print("\nStarting from:")
print("epoch = "+str(epoch))
print("datasetNum = "+str(datasetNum))
print("batchNum = "+str(batchNum)+"\n")      
while(epoch < num_epochs):    
    while(exists("datasets/Train_Datasets/dataset_"+str(datasetNum)+".txt")):
        dataset = datasetHandler.LSTMdataset("datasets/Train_Datasets", datasetNum)
        train_loader = DataLoader(dataset=dataset,
                              batch_size=model.batch_size,
                              shuffle=False,
                              num_workers=0)

        #total_samples = len(dataset)
        #n_iterations = math.ceil(total_samples/model.batch_size)

        ##########################################################

        for i, (inputs, labels) in enumerate(train_loader):
            if i == batchNum:
                # Forward pass and loss
                y_pred = model(inputs)
                y_pred = y_pred.view(y_pred.size(0))
                
                labels = labels.view(labels.size(0))
                #labels = labels.long()
                loss = criterion(y_pred, labels)
                if datasetNum == printingDataset and batchNum == printingBatch:
                    print("Epoch : "+str(epoch)+"  Loss : "+str(loss.item()))
                    print("")
                
                # Backward pass and update
                loss.backward()
                optimizer.step()  
                              
                # zero grad before new step
                optimizer.zero_grad()
                
                batchNum += 1

        ##########################################################
        
        datasetNum += 1
        batchNum = 0
    epoch += 1
    datasetNum = 1

signal_handler(0, 0)