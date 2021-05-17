import argparse
import pandas as pd
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
from torch.autograd import Variable
import time
from datetime import datetime
import torch.optim as optim
import os
from DataMatrix import DataMatrices
from model_cls.LSTMTagger import LSTMTagger
# from loss import SimpleLossCompute, SimpleLossCompute_tst, Batch_Loss, Test_Loss
torch.set_printoptions(profile="full")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def train_net(total_step, model, loss_function, optimizer, DM, model_dir, model_index):
    max_acc = 0
    for epoch in range(total_step):
        step = 0
        for data_batch in DM:

            step = step + 1

            batch_input = data_batch['data_cls']
            batch_label = data_batch['trend']

            X = torch.tensor(batch_input, dtype=torch.float).cuda()
            batch_size = X.size()[0]
            feature_size = X.size()[1]
            asset_size = X.size()[2]
            window_size = X.size()[3]
            
            X = X.permute((3, 0, 2, 1)).contiguous()
            X = X.view(window_size, batch_size*asset_size, feature_size)

            Y = torch.tensor(batch_label, dtype=torch.long).cuda()
            Y = Y.view(batch_size*asset_size)
            ones = torch.ones(batch_size*asset_size).cuda()

            model.train()
            model.zero_grad()

            model.hidden = model.init_hidden()
            tags = model(X)

            prediction = tags.argmax(dim = 1)
            correct = (prediction == Y).sum().float()
            total = len(Y)
            acc = correct / total
            one_percent = (ones == prediction).sum().float() / total

            loss = loss_function(tags, Y)
            loss.backward()
            optimizer.step()
            if (step) % 20 == 0:
                print("Step: %d| Loss per batch: %f | Accuary: %f | One_percent: %f \n" %
                    (step, loss.item(), acc, one_percent))

        val_acc, one_percent = val_net(model, loss, DM)
        print('-'*50)
        print("Test Epoch: %d | Accuary: %f | One_percent: %f \n " %
                (epoch, val_acc, one_percent))
        print('-'*50)
        if(val_acc > max_acc):
            max_acc = val_acc
            torch.save(model, model_dir+'/'+str(model_index)+"_cls.pkl")


def val_net(model, loss, DM):
    batch = DM.get_test_set()
    batch_input = batch["data_cls"] 
    batch_label = batch["trend"]
    X = torch.tensor(batch_input, dtype=torch.float).cuda()  
    X = torch.tensor(batch_input, dtype=torch.float).cuda()
    batch_size = X.size()[0]
    feature_size = X.size()[1]
    asset_size = X.size()[2]
    window_size = X.size()[3]
    
    X = X.permute((3, 0, 2, 1)).contiguous()
    X = X.view(window_size, batch_size*asset_size, feature_size)

    Y = torch.tensor(batch_label, dtype=torch.long).cuda()
    Y = Y.view(batch_size*asset_size)
    ones = torch.ones(batch_size*asset_size).cuda()

    model.hidden = model.init_hidden()
    tags = model(X)
    prediction = tags.argmax(dim = 1)

    correct = (prediction == Y).sum().float()
    total = len(Y)
    acc = correct / total
    one_percent = (ones == prediction).sum().float() / total

    # loss = loss_function(tags, Y)
    return acc, one_percent


def main(epoch=800, batch_size=16, window_size=30, trend_size=1, coin_number=10, feature_number=5,
          test_portion=0.15, portion_reversed=False, is_permed=True, buffer_bias_ratio=5e-5, lr=0.01):
    model_dir = './checkpoint_cls'
    model_index = 'LSTM_1'
    
    DM = DataMatrices(batch_size=batch_size,
                      window_size=window_size,
                      coin_number=coin_number,
                      feature_number=feature_number,
                      test_portion=test_portion,
                      trend_size=trend_size,
                      portion_reversed=portion_reversed,
                      is_permed=is_permed,
                      buffer_bias_ratio=buffer_bias_ratio)
    
    model = LSTMTagger(4, 128, window_size, batch_size, coin_number).cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    train_net(epoch, model, loss_function, optimizer, DM,
          model_dir, model_index)


if __name__ == '__main__':
    main()




