import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMTagger(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, window_size, batch_size, asset_size):
        super(LSTMTagger, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.batch_size = batch_size
        self.asset_size = asset_size

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(256)
        self.norm3 = nn.BatchNorm1d(512)
        # self.out2tag = nn.Linear(512, 2)
        self.out2tag = nn.Linear(self.hidden_dim, 2)
        self.hidden = self.init_hidden()

    def init_hidden(self, train=True):
        # 各个维度的含义是 (Seguence, minibatch_size, hidden_dim)
        return (torch.randn(1, self.batch_size*self.asset_size, self.hidden_dim).cuda(),
                torch.randn(1, self.batch_size*self.asset_size, self.hidden_dim).cuda())

    def forward(self, inputs):
        # 各个维度的含义是 (Seguence, minibatch_size, hidden_dim)
        window_s = inputs.size()[0]
        batch_asset = inputs.size()[1]

        # out, (hn, cn) = self.lstm(inputs, self.hidden)
        out, (hn, cn) = self.lstm(inputs)
        # out = F.dropout(out, 0.5)
        # out = F.relu(out)

        # out = out.permute((1, 0, 2)).contiguous()
        out = out[-1,:,  :]
        #out = self.fc1(out)
        # out = self.norm1(out)
        #out = F.dropout(F.relu(out), 0.2)
        #out = self.fc2(out)
        # out = self.norm2(out)
        #out = F.dropout(F.relu(out), 0.2)
        #out = self.fc3(out)
        # out = self.norm3(out)
        #out = F.dropout(F.relu(out), 0.2)

        # out(128*11,2)
        out = self.out2tag(out)

        # tags = F.softmax(out)
        return out