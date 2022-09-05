import datasetHandler
import VS_LSTM
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
    "batch_number" : batchNum,
    }
    FILE = "Exercises/"+str(exercise)+"/checkpoint.pth"
    torch.save(checkpoint, FILE)
    
    print('\nExiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

exercise = int(input("Enter the exercise number: "))

with open("Exercises/"+str(exercise)+"/Settings.txt", 'r') as s:
    for line in s:
        line = line.split("\n")[0]
        data = line.split(" = ")[1]
        tag = line.split(" = ")[0]
        if tag == "learning_rate":
            learning_rate = float(data)
        elif tag == "num_epochs":
            num_epochs = int(data)
        elif tag == "num_layers":
            num_layers = int(data)
        elif tag == "batchSize":
            batchSize = int(data)
        elif tag == "printingBatch":
            printingBatch = int(data)

#Settings:
criterion = nn.MSELoss()
dataset = datasetHandler.LSTMdataset("Exercises/"+str(exercise), "train_dataset.txt")
#total_samples = len(dataset)
#n_iterations = math.ceil(total_samples/batchSize)
inputSize = len(dataset[0][0][0])
model = VS_LSTM.LSTM(num_layers, inputSize*2, inputSize)
#model = LSTM.LSTM(1, len(dataset[0][0]), inputSize)
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

epoch = 0
batchNum = 0
loss = "Dummy Initialisation"

ch = input("Use existing datasets? Y/N: ")
if ch.upper() == "N":
    ds = datasetHandler.datasetHandler()
    ds.createDataset(exercise)
    ds.shuffleDataset("Exercises/"+str(exercise), "master_dataset.txt")
    ds.splitDataset("Exercises/"+str(exercise), "master_dataset.txt")
    
    with open("Exercises/"+str(exercise)+"/checkpoint.pth", "w") as c:
        pass
else:
    ch1 = input("Start training from scratch? Y/N: ")
    if ch1.upper() == "Y":
        with open("Exercises/"+str(exercise)+"/checkpoint.pth", "w") as c:
            pass
    else:
        FILE = "Exercises/"+str(exercise)+"/checkpoint.pth"
        checkpoint = torch.load(FILE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        epoch = checkpoint['epoch']
        batchNum = checkpoint['batch_number']
        with open("Exercises/"+str(exercise)+"/checkpoint.pth", "w") as c:
            pass

print("\nStarting from:")
print("epoch = "+str(epoch))
print("batchNum = "+str(batchNum))
print("batchSize = "+str(batchSize)+"\n")

train_loader = DataLoader(dataset=dataset,
                      batch_size=batchSize,
                      shuffle=False,
                      num_workers=0)      

while(epoch < num_epochs):    

    ##########################################################
    for i, (inputs, labels, seqLens) in enumerate(train_loader):
        if i == batchNum:
            seqLens = seqLens.view(seqLens.size(0))
            seqLens = [int(x) for x in seqLens]
            # Forward pass and loss
            y_pred = model(inputs, seqLens)
            
            #y_pred = model(inputs)
            #y_pred = y_pred.view(y_pred.size(0))
            
            labels = labels.view(labels.size(0))
            #labels = labels.long()
            
            loss = criterion(y_pred, labels)
            if batchNum == printingBatch:
                print("Epoch : "+str(epoch)+"  BatchNum : "+str(i)+"  Loss : "+str(loss.item()))
                print("")
                print("y_pred:")
                print(y_pred)
                print("")
                print("labels:")
                print(labels)
                print("\n")
            
            
            # Backward pass and update
            loss.backward()
            optimizer.step()  
                          
            # zero grad before new step
            optimizer.zero_grad()
            
            batchNum += 1

    ##########################################################
    
    batchNum = 0
    epoch += 1

signal_handler(0, 0)
