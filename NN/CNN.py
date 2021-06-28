import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_class, drop_prob):
        super(CNN, self).__init__()
        # input is 28x28
        self.dropout = nn.Dropout(p=drop_prob)
        self.conv1 = nn.Conv2d(1, 32, 5, padding = 2)  #(input_channel, output_channel, filter_size, padding_size)
        self.conv2 = nn.Conv2d(32, 32, 5, padding = 2)
        '''
        add here
        '''
        self.fc1 = nn.Linear(32*7*7, 1024)  #28/2/2 -> pooling_stride = 2 && 2 times
        self.reduce_layer = nn.Linear(1024, num_class)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  #(batch_size, 32, 14, 14)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        '''
        add here
        '''
        x = x.view(-1, 32*7*7)  #flatten
        x = self.dropout(F.relu(self.fc1(x)))
        output = self.reduce_layer(x)
        
        return self.logsoftmax(output)