import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)  #out : (batch_size, sequence_len, hidden_size)
        out = self.fc(out[:, -1, :])
        
        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  #out : (batch_size, sequence_len, hidden_size)
        out = self.fc(out[:, -1, :])
        
        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)  #out : (batch_size, sequence_len, hidden_size)
        out = self.fc(out[:, -1, :])

        return out