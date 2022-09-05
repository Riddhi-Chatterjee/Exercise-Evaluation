import signal
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

def signal_handler(sig, frame):
    print('\nExiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class LSTM(nn.ModuleList):
    def __init__(self, numClasses, seqLen, inputSize):
        super(LSTM, self).__init__()      
              
        self.MAX_X_LENGTH = inputSize
        
        # Number of samples per time step:
        self.batch_size = 720         
        
        # Dimension of hidden states:
        factor = (self.MAX_X_LENGTH - numClasses)/4
        self.hidden_dim_1 = self.MAX_X_LENGTH
        self.hidden_dim_2 = int(self.MAX_X_LENGTH - factor)
        self.hidden_dim_3 = int(self.MAX_X_LENGTH - (2*factor))
        self.hidden_dim_4 = int(self.MAX_X_LENGTH - (3*factor))
        #self.hidden_dim_5 = int(self.MAX_X_LENGTH - (4*factor))
        
        # Input size
        self.inputSize = self.MAX_X_LENGTH          
               
        # Number of time steps
        self.seqLen = seqLen           
        
        # Initialising LSTM Cell for the first layer
        self.lstm_cell_layer_1 = nn.LSTMCell(self.inputSize, self.hidden_dim_1)

        # Initialising LSTM Cell for the second layer
        self.lstm_cell_layer_2 = nn.LSTMCell(self.hidden_dim_1, self.hidden_dim_2)
        
        # Initialising LSTM Cell for the third layer
        self.lstm_cell_layer_3 = nn.LSTMCell(self.hidden_dim_2, self.hidden_dim_3)
        
        # Initialising LSTM Cell for the fourth layer
        self.lstm_cell_layer_4 = nn.LSTMCell(self.hidden_dim_3, self.hidden_dim_4)
        
        # Initialising LSTM Cell for the fifth layer
        #self.lstm_cell_layer_5 = nn.LSTMCell(self.hidden_dim_4, self.hidden_dim_5)
        
        #Creating a fully-connected layer
        self.fully_connected = nn.Linear(self.hidden_dim_4, numClasses)
        
        #Sigmoid layer:
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        hidden_state_1 = torch.rand(x.size(0), self.hidden_dim_1)
        cell_state_1 = torch.rand(x.size(0), self.hidden_dim_1)
        hidden_state_2 = torch.rand(x.size(0), self.hidden_dim_2)
        cell_state_2 = torch.rand(x.size(0), self.hidden_dim_2)
        hidden_state_3 = torch.rand(x.size(0), self.hidden_dim_3)
        cell_state_3 = torch.rand(x.size(0), self.hidden_dim_3)
        hidden_state_4 = torch.rand(x.size(0), self.hidden_dim_4)
        cell_state_4 = torch.rand(x.size(0), self.hidden_dim_4)
        #hidden_state_5 = torch.rand(x.size(0), self.hidden_dim_5)
        #cell_state_5 = torch.rand(x.size(0), self.hidden_dim_5)
        
        # state initialisation
        torch.nn.init.xavier_normal_(hidden_state_1)
        torch.nn.init.xavier_normal_(cell_state_1)
        torch.nn.init.xavier_normal_(hidden_state_2)
        torch.nn.init.xavier_normal_(cell_state_2)
        torch.nn.init.xavier_normal_(hidden_state_3)
        torch.nn.init.xavier_normal_(cell_state_3)
        torch.nn.init.xavier_normal_(hidden_state_4)
        torch.nn.init.xavier_normal_(cell_state_4)
        #torch.nn.init.xavier_normal_(hidden_state_5)
        #torch.nn.init.xavier_normal_(cell_state_5)
            
        # Prepare the shape for LSTMCell
        out = x.transpose(0,1).view(self.seqLen, x.size(0), -1) 
               
        # Unfolding LSTM
        # Last hidden_state will be used to feed the fully connected neural net
        for i in range(self.seqLen):
            hidden_state_1, cell_state_1 = self.lstm_cell_layer_1(out[i], (hidden_state_1, cell_state_1))
            hidden_state_2, cell_state_2 = self.lstm_cell_layer_2(hidden_state_1, (hidden_state_2, cell_state_2))   
            hidden_state_3, cell_state_3 = self.lstm_cell_layer_3(hidden_state_2, (hidden_state_3, cell_state_3))
            hidden_state_4, cell_state_4 = self.lstm_cell_layer_4(hidden_state_3, (hidden_state_4, cell_state_4))
            #hidden_state_5, cell_state_5 = self.lstm_cell_layer_5(hidden_state_4, (hidden_state_5, cell_state_5))
              
        # Last hidden state is passed through a fully connected neural net
        out = self.fully_connected(hidden_state_4)
        out = (self.sigmoid(out) * 40) + 10	        
        return out