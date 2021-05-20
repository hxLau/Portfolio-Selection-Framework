import argparse
import pandas as pd
import sqlite3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.autograd import Variable
import os
import model_ps.AdvRAT as AdvRAT
from DataMatrix import DataMatrices
from loss import SimpleLossCompute, SimpleLossCompute_tst, Batch_Loss, Test_Loss, SimpleAdvLossCompute, CRRA_Loss
from util.GaussianNoise import *
from util.tool import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_parameter():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--total_step', type=int, default=80000)
    parser.add_argument('--x_window_size', type=int, default=31)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--coin_num', type=int, default=11)
    parser.add_argument('--feature_number', type=int, default=4)
    parser.add_argument('--output_step', type=int, default=1000)
    parser.add_argument('--model_index', type=int, default=0)
    parser.add_argument('--multihead_num', type=int, default=2)
    parser.add_argument('--local_context_length', type=int, default=5)
    parser.add_argument('--model_dim', type=int, default=12)


    parser.add_argument('--test_portion', type=float, default=0.08)
    parser.add_argument('--trading_consumption', type=float, default=0.0025)
    parser.add_argument('--gamma', type=float, default=5.0)
    parser.add_argument('--cost_penalty', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-8)
    parser.add_argument('--daily_interest_rate', type=float, default=0.001)

    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--model_dir', type=str, default='./checkpoint')
    parser.add_argument('--adv', type=int, default=1)  # 1:adversarial training   0: normal training
    parser.add_argument('--adv_local', type=int, default=1) # 1: decoder feature   2: encoder feature   3: input feature
    parser.add_argument('--gaussion_noise', type=int, default=0)    # 1: noise    0: no noise
    parser.add_argument('--utility_function', type=int, default=1) # 1: log utility   2: CRRA
    parser.add_argument('--market', type=str, default='CC1')

    FLAGS = parser.parse_args()
    return FLAGS

def get_csv_name(adv, adv_local, coin_num, gaussion_noise, utility_function, gamma, market):
    csv_name = ''
    if adv:
        csv_name = csv_name + 'AdvRAT'
        if adv_local==1:
            csv_name = csv_name + '_IN'
        elif adv_local==2:
            csv_name = csv_name + '_EN'
        elif adv_local==3:
            csv_name = csv_name + '_DE'
    else:
        csv_name = csv_name + 'RAT'

    csv_name = csv_name + '_' + market + '_' + str(coin_num)

    if gaussion_noise:
        csv_name = csv_name + '_NOISE'

    if utility_function==1:
        csv_name = csv_name + '_LOG'
    else:
        csv_name = csv_name + '_CRRA' + str(gamma)

    print(csv_name)
    return csv_name

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if self.warmup == 0:
            return self.factor
        else:
            return self.factor * \
                (self.model_size ** (-0.5) *
                 min(step ** (-0.5), step * self.warmup ** (-1.5)))


def subsequent_mask(size):  
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(
        np.ones(attn_shape), k=1).astype('uint8') 
    return torch.from_numpy(subsequent_mask) == 0  


def make_std_mask(local_price_context, batch_size):
    "Create a mask to hide padding and future words."
    local_price_mask = (torch.ones(batch_size, 1, 1) == 1)
    local_price_mask = local_price_mask & (subsequent_mask(
        local_price_context.size(-2)).type_as(local_price_mask.data))
    return local_price_mask


def test_batch(DM, x_window_size, model, evaluate_loss_compute, local_context_length):
    tst_batch = DM.get_test_set()
    tst_batch_input = tst_batch["data_ps"]  
    tst_batch_y = tst_batch["relative_price"]
    tst_batch_last_w = tst_batch["last_w"]
    tst_batch_w = tst_batch["setw"]

    tst_previous_w = torch.tensor(tst_batch_last_w, dtype=torch.float).cuda()
    tst_previous_w = torch.unsqueeze(tst_previous_w, 1)  
    tst_batch_input = tst_batch_input.transpose((1, 0, 2, 3))
    tst_batch_input = tst_batch_input.transpose((0, 1, 3, 2))
    tst_src = torch.tensor(tst_batch_input, dtype=torch.float).cuda()

    tst_src_mask = (torch.ones(tst_src.size()[1], 1, x_window_size) == 1)
    tst_currt_price = tst_src.permute(
        (3, 1, 2, 0))  

    if(local_context_length > 1):
        padding_price = tst_currt_price[:,
                                        :, -(local_context_length)*2+1:-1, :]
    else:
        padding_price = None


    tst_currt_price = tst_currt_price[:, :, -1:, :]
    tst_trg_mask = make_std_mask(tst_currt_price, tst_src.size()[1])
    tst_batch_y = tst_batch_y.transpose((0, 2, 1))  
    tst_trg_y = torch.tensor(tst_batch_y, dtype=torch.float).cuda()

    tst_out, input_feature, encoder_feature, decoder_feature = model.forward(tst_src, tst_currt_price, tst_previous_w,  
                            tst_src_mask, tst_trg_mask, padding_price)

    tst_loss, tst_portfolio_value = evaluate_loss_compute(tst_out, tst_trg_y)
    return tst_loss, tst_portfolio_value


def test_online(DM, x_window_size, model, evaluate_loss_compute, local_context_length):
    tst_batch = DM.get_test_set_online()
    tst_batch_input = tst_batch["data_ps"]
    tst_batch_y = tst_batch["relative_price"]
    tst_batch_last_w = tst_batch["last_w"]
    tst_batch_w = tst_batch["setw"]

    tst_previous_w = torch.tensor(tst_batch_last_w, dtype=torch.float).cuda()
    tst_previous_w = torch.unsqueeze(tst_previous_w, 1)

    tst_batch_input = tst_batch_input.transpose((1, 0, 2, 3))
    tst_batch_input = tst_batch_input.transpose((0, 1, 3, 2))

    long_term_tst_src = torch.tensor(tst_batch_input, dtype=torch.float).cuda()

    tst_src_mask = (torch.ones(long_term_tst_src.size()
                               [1], 1, x_window_size) == 1)

    long_term_tst_currt_price = long_term_tst_src.permute((3, 1, 2, 0))
    long_term_tst_currt_price = long_term_tst_currt_price[:,
                                                          :, x_window_size-1:, :]

    tst_trg_mask = make_std_mask(
        long_term_tst_currt_price[:, :, 0:1, :], long_term_tst_src.size()[1])

    tst_batch_y = tst_batch_y.transpose((0, 3, 2, 1))
    tst_trg_y = torch.tensor(tst_batch_y, dtype=torch.float).cuda()
    tst_long_term_w = []
    tst_y_window_size = len(DM._test_ind)-x_window_size-1-1
    for j in range(tst_y_window_size+1):  
        tst_src = long_term_tst_src[:, :, j:j+x_window_size, :]
        tst_currt_price = long_term_tst_currt_price[:, :, j:j+1, :]
        if(local_context_length > 1):
            padding_price = long_term_tst_src[:, :, j+x_window_size -
                                              1-local_context_length*2+2:j+x_window_size-1, :]
            padding_price = padding_price.permute(
                (3, 1, 2, 0))  
        else:
            padding_price = None
        out, input_feature, encoder_feature, decoder_feature = model.forward(tst_src, tst_currt_price, tst_previous_w,  
                            tst_src_mask, tst_trg_mask, padding_price)
        if(j == 0):
            tst_long_term_w = out.unsqueeze(0)  
        else:
            tst_long_term_w = torch.cat([tst_long_term_w, out.unsqueeze(0)], 0)
        out = out[:, :, 1:]  
        tst_previous_w = out
    tst_long_term_w = tst_long_term_w.permute(
        1, 0, 2, 3)  
    tst_loss, tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO = evaluate_loss_compute(
        tst_long_term_w, tst_trg_y)
    return tst_loss, tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO


def test_net(DM, total_step, output_step, x_window_size, local_context_length, model, loss_compute, evaluate_loss_compute, is_trn=True, evaluate=True):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    max_tst_portfolio_value = 0

    for i in range(total_step):
        if(is_trn):
            loss, portfolio_value = train_one_step(
                DM, x_window_size, model, loss_compute, local_context_length)
            total_loss += loss.item()
        if (i % output_step == 0 and is_trn):
            elapsed = time.time() - start
            print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f | batch per Sec: %f \r\n" %
                  (i, loss.item(), portfolio_value.item(), output_step / elapsed))
            start = time.time()

        tst_total_loss = 0
        with torch.no_grad():
            if(i % output_step == 0 and evaluate):
                model.eval()
                tst_loss, tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO = test_online(
                    DM, x_window_size, model, evaluate_loss_compute, local_context_length)
                tst_total_loss += tst_loss.item()
                elapsed = time.time() - start
                print("Test: %d Loss: %f| Portfolio_Value: %f | SR: %f | CR: %f | TO: %f |testset per Sec: %f" %
                      (i, tst_loss.item(), tst_portfolio_value.item(), SR.item(), CR.item(), TO.item(), 1/elapsed))
                start = time.time()


                if(tst_portfolio_value > max_tst_portfolio_value):
                    max_tst_portfolio_value = tst_portfolio_value
                    log_SR = SR
                    log_CR = CR
                    log_St_v = St_v
                    log_tst_pc_array = tst_pc_array
    return max_tst_portfolio_value, log_SR, log_CR, log_St_v, log_tst_pc_array, TO


def train_one_step(DM, x_window_size, model, loss_compute, local_context_length, opt, adv=True, adv_local=1, gaussion_noise=False):
    batch = DM.next_batch()
    batch_input = batch["data_ps"]  
    batch_y = batch["relative_price"]  
    batch_last_w = batch["last_w"]  
    batch_w = batch["setw"]

    if gaussion_noise:
        np_input = np.array(batch_input)
        batch_input = add_noise(np_input)
        
    previous_w = torch.tensor(batch_last_w, dtype=torch.float).cuda()
    previous_w = torch.unsqueeze(previous_w, 1)  
    batch_input = batch_input.transpose((1, 0, 2, 3))
    batch_input = batch_input.transpose((0, 1, 3, 2))
    src = torch.tensor(batch_input, dtype=torch.float).cuda()  

    price_series_mask = (torch.ones(src.size()[1], 1, x_window_size) == 1)
    currt_price = src.permute((3, 1, 2, 0))  
    if(local_context_length > 1):
        padding_price = currt_price[:, :, -(local_context_length)*2+1:-1, :]
    else:
        padding_price = None
    currt_price = currt_price[:, :, -1:, :]  

    trg_mask = make_std_mask(currt_price, src.size()[1])  
    batch_y = batch_y.transpose((0, 2, 1))  
    trg_y = torch.tensor(batch_y, dtype=torch.float).cuda()
    out, input_feature, encoder_feature, decoder_feature = model.forward(src, currt_price, previous_w,
                        price_series_mask, trg_mask, padding_price)
    new_w = out[:, :, 1:]  
    new_w = new_w[:, 0, :]  
    new_w = new_w.detach().cpu().numpy()
    batch_w(new_w)

    loss, portfolio_value = loss_compute(out, trg_y)
    if adv:
      if adv_local==1:
        advgrad = torch.autograd.grad(loss, input_feature, retain_graph=True)
        adv_input_feature = input_feature + 0.05*advgrad[0]
        advout = model.advforward_in(adv_input_feature, currt_price, previous_w,
                            price_series_mask, trg_mask, padding_price)
       
      elif adv_local==2:
        advgrad = torch.autograd.grad(loss, encoder_feature, retain_graph=True)
        adv_encoder_feature = encoder_feature + 0.05*advgrad[0]
        advout = model.advforward_en(adv_encoder_feature, currt_price, previous_w,
                            price_series_mask, trg_mask, padding_price)
      
      elif adv_local==3:
        advgrad = torch.autograd.grad(loss, decoder_feature, retain_graph=True)
        adv_decoder_feature = decoder_feature + 0.05*advgrad[0]
        advout = model.advforward_de(adv_decoder_feature, previous_w)

      else:
        print('adv_local must be 1 or 2 or 3')

      advloss, advportfolio_value = loss_compute(advout, trg_y)
      loss = loss + 0.05*advloss
    loss.backward()
    opt.step()
    opt.optimizer.zero_grad()

    return loss, portfolio_value


def train_net(DM, total_step, output_step, x_window_size, local_context_length, model, model_dir, model_index, loss_compute, 
              evaluate_loss_compute, opt, is_trn=True, evaluate=True, adv=True, adv_local=1, gaussion_noise=False):
    "Standard Training and Logging Function"
    start = time.time()

    total_loss = 0
    max_tst_portfolio_value = 0
    for i in range(total_step):
        if(is_trn):
            model.train()
            loss, portfolio_value = train_one_step(
                DM, x_window_size, model, loss_compute, local_context_length, opt, adv, adv_local, gaussion_noise=False)
            total_loss += loss.item()
        if (i % output_step == 0 and is_trn):
            elapsed = time.time() - start
            print("Epoch Step: %d| Loss per batch: %f| Portfolio_Value: %f | batch per Sec: %f \r\n" %
                  (i, loss.item(), portfolio_value.item(), output_step / elapsed))
            start = time.time()

        tst_total_loss = 0
        with torch.no_grad():
            if(i % output_step == 0 and evaluate):
                model.eval()
                tst_loss, tst_portfolio_value = test_batch(
                    DM, x_window_size, model, evaluate_loss_compute, local_context_length)

                tst_total_loss += tst_loss.item()
                elapsed = time.time() - start
                print("Test: %d Loss: %f| Portfolio_Value: %f | testset per Sec: %f \r\n" %
                      (i, tst_loss.item(), tst_portfolio_value.item(), 1/elapsed))
                start = time.time()

                if(tst_portfolio_value > max_tst_portfolio_value):
                    max_tst_portfolio_value = tst_portfolio_value
                    torch.save(model, model_dir+'/'+str(model_index)+".pkl")
                    print("save model!")
    return tst_loss, tst_portfolio_value


def main():
    FLAGS = get_parameter()
    lr_model_sz = 5120
    factor = FLAGS.learning_rate 
    warmup = 0  
    total_step = FLAGS.total_step
    x_window_size = FLAGS.x_window_size  
    batch_size = FLAGS.batch_size
    coin_num = FLAGS.coin_num  
    feature_number = FLAGS.feature_number  
    trading_consumption = FLAGS.trading_consumption  
    gamma = FLAGS.gamma
    cost_penalty = FLAGS.cost_penalty  
    output_step = FLAGS.output_step  
    local_context_length = FLAGS.local_context_length
    model_dim = FLAGS.model_dim
    weight_decay = FLAGS.weight_decay
    interest_rate = FLAGS.daily_interest_rate/24/2
    adv = FLAGS.adv
    adv_local = FLAGS.adv_local
    gaussion_noise = FLAGS.gaussion_noise
    utility_function = FLAGS.utility_function
    market = FLAGS.market
    test_portion = FLAGS.test_portion

    csv_name = get_csv_name(adv, adv_local, coin_num, gaussion_noise, utility_function, gamma, market)

    DM = DataMatrices(batch_size=batch_size, window_size=x_window_size, coin_number=coin_num, feature_number=feature_number,
                      test_portion=test_portion, trend_size=1, portion_reversed=False, is_permed=True,
                      buffer_bias_ratio=5e-5, market=market, picture_bool=False, predict_bool=False)

    model = AdvRAT.make_model(batch_size, coin_num, x_window_size, feature_number-1,
                        N=1, d_model_Encoder=FLAGS.multihead_num*model_dim,
                        d_model_Decoder=FLAGS.multihead_num*model_dim,
                        d_ff_Encoder=FLAGS.multihead_num*model_dim,
                        d_ff_Decoder=FLAGS.multihead_num*model_dim,
                        h=FLAGS.multihead_num,
                        dropout=0.01,
                        local_context_length=local_context_length)
    model = model.cuda()

    model_opt = NoamOpt(lr_model_sz, factor, warmup, torch.optim.Adam(
        model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay))

    if utility_function==2:
        loss_compute = SimpleAdvLossCompute(CRRA_Loss(
            trading_consumption, interest_rate, gamma, True), model_opt)
        evaluate_loss_compute = SimpleLossCompute(CRRA_Loss(
            trading_consumption, interest_rate, gamma, False),  None)
    else:
        loss_compute = SimpleAdvLossCompute(Batch_Loss(
            trading_consumption, interest_rate, gamma, True), model_opt)
        evaluate_loss_compute = SimpleLossCompute(Batch_Loss(
            trading_consumption, interest_rate, gamma, False),  None)

    test_loss_compute = SimpleLossCompute_tst(Test_Loss(
        trading_consumption, interest_rate, gamma, False),  None)


    ##########################train net####################################################
    tst_loss, tst_portfolio_value = train_net(DM, total_step, output_step, x_window_size, local_context_length,
                                            model, FLAGS.model_dir, FLAGS.model_index, loss_compute, evaluate_loss_compute, model_opt, True, True,adv, adv_local, gaussion_noise)

    model = torch.load(FLAGS.model_dir+'/' + str(FLAGS.model_index)+'.pkl')

    ##########################test net#####################################################
    tst_portfolio_value, SR, CR, St_v, tst_pc_array, TO = test_net(
        DM, 1, 1, x_window_size, local_context_length, model, loss_compute, test_loss_compute, False, True)

    csv_dir = FLAGS.log_dir+"/"+ csv_name +".csv"
    d = {"net_dir": [FLAGS.model_index],
        "fAPV": [tst_portfolio_value.item()],
        "SR": [SR.item()],
        "CR": [CR.item()],
        "TO": [TO.item()],
        "St_v": [''.join(str(e)+', ' for e in St_v)],
        "backtest_test_history": [''.join(str(e)+', ' for e in tst_pc_array.cpu().numpy())],
        }
    new_data_frame = pd.DataFrame(data=d).set_index("net_dir")
    if os.path.isfile(csv_dir):
        dataframe = pd.read_csv(csv_dir).set_index("net_dir")
        dataframe = dataframe.append(new_data_frame)
    else:
        dataframe = new_data_frame
    dataframe.to_csv(csv_dir)



if __name__ == '__main__':
    main()
