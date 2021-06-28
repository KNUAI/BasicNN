import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_class, drop_prob):
        super(MLP, self).__init__()
        # input is 28x28
        # need for flatten ==> 784
        self.dropout = nn.Dropout(p=drop_prob)
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 512)
        '''
        add here
        '''
        self.reduce_layer = nn.Linear(512, num_class)
        self.logsoftmax = nn.LogSoftmax(dim=1)
       
    def forward(self, x):
        # need for flatten ==> 784
        x = self.dropout(F.relu(self.linear1(x.view(-1, 784))))
        x = self.dropout(F.relu(self.linear2(x)))
        '''
        add here
        '''
        output = self.reduce_layer(x)

        return self.logsoftmax(output)