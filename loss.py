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


class Batch_Loss(nn.Module):
    def __init__(self, commission_ratio, interest_rate, gamma=0.1, size_average=True):
        super(Batch_Loss, self).__init__()
        self.gamma = gamma  
        self.size_average = size_average
        self.commission_ratio = commission_ratio
        self.interest_rate = interest_rate

    def forward(self, w, y):         
        close_price = y[:, :, 0:1].cuda()  
        # future close prise (including cash)
        close_price = torch.cat([torch.ones(close_price.size()[0], 1, 1).cuda(
        ), close_price], 1).cuda()  
        reward = torch.matmul(w, close_price)  
        close_price = close_price.view(close_price.size()[0], close_price.size()[
                                       2], close_price.size()[1]) 
###############################################################################################################
        element_reward = w*close_price
        interest = torch.zeros(element_reward.size(), dtype=torch.float).cuda()
        interest[element_reward < 0] = element_reward[element_reward < 0]
        interest = torch.sum(interest, 2).unsqueeze(2) * \
            self.interest_rate 
###############################################################################################################
        future_omega = w*close_price/reward 
        wt = future_omega[:-1] 
        wt1 = w[1:]  
        pure_pc = 1-torch.sum(torch.abs(wt-wt1), -1) * \
            self.commission_ratio  
        pure_pc = pure_pc.cuda()
        pure_pc = torch.cat([torch.ones([1, 1]).cuda(), pure_pc], 0)
        pure_pc = pure_pc.view(
            pure_pc.size()[0], 1, pure_pc.size()[1])  

################## Deduct transaction fee ##################
        reward = reward*pure_pc  # reward=pv_vector
################## Deduct loan interest ####################
        reward = reward+interest
        portfolio_value = torch.prod(reward, 0)
        batch_loss = -torch.log(reward)

        loss = batch_loss.mean()
        return loss, portfolio_value[0][0]

def util(r, gamma=5.0):
  util_power_coeff = 1.0 - gamma
  return (torch.pow( r, util_power_coeff ) - 1.0) / util_power_coeff

class CRRA_Loss(nn.Module):
    def __init__(self, commission_ratio, interest_rate, gamma=0.1, size_average=True):
        super(CRRA_Loss, self).__init__()
        self.gamma = gamma  # variance penalty
        self.size_average = size_average
        self.commission_ratio = commission_ratio
        self.interest_rate = interest_rate

    def forward(self, w, y):   
        close_price = y[:, :, 0:1].cuda() 
        # future close prise (including cash)
        close_price = torch.cat([torch.ones(close_price.size()[0], 1, 1).cuda(
        ), close_price], 1).cuda() 
        reward = torch.matmul(w, close_price) 
        close_price = close_price.view(close_price.size()[0], close_price.size()[
                                       2], close_price.size()[1])
###############################################################################################################
        element_reward = w*close_price
        interest = torch.zeros(element_reward.size(), dtype=torch.float).cuda()
        interest[element_reward < 0] = element_reward[element_reward < 0]
        interest = torch.sum(interest, 2).unsqueeze(2) * \
            self.interest_rate 
###############################################################################################################
        future_omega = w*close_price/reward
        wt = future_omega[:-1]
        wt1 = w[1:] 

        pure_pc = 1-torch.sum(torch.abs(wt-wt1), -1) * \
            self.commission_ratio 
        pure_pc = pure_pc.cuda()
        pure_pc = torch.cat([torch.ones([1, 1]).cuda(), pure_pc], 0)
        pure_pc = pure_pc.view(
            pure_pc.size()[0], 1, pure_pc.size()[1]) 

################## Deduct transaction fee ##################
        reward = reward*pure_pc  # reward=pv_vector
################## Deduct loan interest ####################
        reward = reward+interest
        portfolio_value = torch.prod(reward, 0)
        crra_loss = -util(reward, self.gamma)
        batch_loss = -torch.log(reward)

        loss = crra_loss.mean()
        return loss, portfolio_value[0][0]


class Pre_Train_Loss(nn.Module):
    def __init__(self, size_average=True):
        super(Pre_Train_Loss, self).__init__()

    def forward(self, predict, y):
        y = y[:, :, :,0:1].cuda()
        y = y.squeeze(-1)
        y = y - predict
        loss = y.mul(y)
        loss = loss.mean()
        return loss


class SimpleAdvLossCompute:
    "A simple loss compute and train function."

    def __init__(self,  criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y):
        loss, portfolio_value = self.criterion(x, y)
        '''
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        '''
        return loss, portfolio_value

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self,  criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y):
        loss, portfolio_value = self.criterion(x, y)
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss, portfolio_value



def max_drawdown(pc_array):
    """calculate the max drawdown with the portfolio changes
    @:param pc_array: all the portfolio changes during a trading process
    @:return: max drawdown
    """
    portfolio_values = []
    drawdown_list = []
    max_benefit = 0
    for i in range(pc_array.shape[0]):
        if i > 0:
            portfolio_values.append(portfolio_values[i - 1] * pc_array[i])
        else:
            portfolio_values.append(pc_array[i])
        if portfolio_values[i] > max_benefit:
            max_benefit = portfolio_values[i]
            drawdown_list.append(0.0)
        else:
            drawdown_list.append(1.0 - portfolio_values[i] / max_benefit)
    return max(drawdown_list)


class Test_Loss(nn.Module):
    def __init__(self, commission_ratio, interest_rate, gamma=0.1, size_average=True):
        super(Test_Loss, self).__init__()
        self.gamma = gamma  
        self.size_average = size_average
        self.commission_ratio = commission_ratio
        self.interest_rate = interest_rate

    def forward(self, w, y):              
        close_price = y[:, :, :, 0:1].cuda() 
        close_price = torch.cat([torch.ones(close_price.size()[0], close_price.size()[
                                1], 1, 1).cuda(), close_price], 2).cuda() 

        reward = torch.matmul(w, close_price)
        close_price = close_price.view(close_price.size()[0], close_price.size()[
                                       1], close_price.size()[3], close_price.size()[2])  
##############################################################################
        element_reward = w*close_price
        interest = torch.zeros(element_reward.size(), dtype=torch.float).cuda()
        interest[element_reward < 0] = element_reward[element_reward < 0]

        interest = torch.sum(interest, 3).unsqueeze(
            3)*self.interest_rate  
##############################################################################

        future_omega = w*close_price/reward
        wt = future_omega[:, :-1]  
        wt1 = w[:, 1:]  
        pure_pc = 1-torch.sum(torch.abs(wt-wt1), -1) * \
            self.commission_ratio  
        pure_pc = pure_pc.cuda()

        pure_pc = torch.cat(
            [torch.ones([pure_pc.size()[0], 1, 1]).cuda(), pure_pc], 1)
        pure_pc = pure_pc.view(pure_pc.size()[0], pure_pc.size()[
                               1], 1, pure_pc.size()[2]) 
        cost_penalty = torch.sum(torch.abs(wt-wt1), -1)  
################## Deduct transaction fee ##################
        reward = reward*pure_pc
################## Deduct loan interest ####################
        reward = reward+interest
        if not self.size_average:
            tst_pc_array = reward.squeeze()
            sr_reward = tst_pc_array-1
            SR = sr_reward.mean()/sr_reward.std()
            SN = torch.prod(reward, 1)
            SN = SN.squeeze()
            St_v = []
            St = 1.
            MDD = max_drawdown(tst_pc_array)
            for k in range(reward.size()[1]):  
                St *= reward[0, k, 0, 0]
                St_v.append(St.item())
            CR = SN/MDD
            TO = cost_penalty.mean()
##############################################
        portfolio_value = torch.prod(reward, 1)
        batch_loss = -torch.log(portfolio_value)

        if self.size_average:
            loss = batch_loss.mean()
            return loss, portfolio_value.mean()
        else:
            loss = batch_loss.mean()
            return loss, portfolio_value[0][0][0], SR, CR, St_v, tst_pc_array, TO


class SimpleLossCompute_tst:
    "A simple loss compute and train function."

    def __init__(self,  criterion, opt=None):
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y):
        if self.opt is not None:
            loss, portfolio_value = self.criterion(x, y)
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
            return loss, portfolio_value
        else:
            loss, portfolio_value, SR, CR, St_v, tst_pc_array, TO = self.criterion(
                x, y)
            return loss, portfolio_value, SR, CR, St_v, tst_pc_array, TO
