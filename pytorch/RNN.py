import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN(nn.Module):
    '''
        Recurrent Neural Network 
    '''
    def __init__(self, input_size, hidden_size, num_layers, output_size, rnn_type: str, drop_out=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_out)
        elif rnn_type == 'LSTM':
            self.rnn =nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_out)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_out)
        else:
            raise ValueError('Rnn Type must be RNN, LSTM, or GRU')
        
        self.fc = nn.Linear(hidden_size, output_size)


    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # Initial hidden state

        if self.rnn_type == 'LSTM':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # Initial cell state
            out, _ = self.rnn(x, (h0, c0))  # LSTM needs both hidden and cell states
        else:
            out, _ = self.rnn(x, h0)  # RNN and GRU only need hidden state

        
        out = self.fc(out[:, -1, :])  # Get the output of the last time step
        return out
    
