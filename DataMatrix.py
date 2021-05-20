from datetime import datetime
import pandas as pd
import numpy as np
import math
import time
import os
from util.Seq2Img import get_image_with_price, image_loader, tensor_to_PIL
from util.PredictLabel import *
from util.Sampling import ReplayBuffer
import matplotlib.pyplot as plt
import torch


def get_type_list(feature_number):
    if feature_number == 1:
        type_list = ['CLOSE']
    elif feature_number == 2:
        type_list = ['CLOSE', 'OPEN']
    elif feature_number == 3:
        type_list = ['CLOSE', 'HIGH', 'LOW']
    elif feature_number == 4:
        type_list = ['CLOSE', 'HIGH', 'LOW', 'OPEN']
    elif feature_number == 5:
        type_list = ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME']
    else:
        raise ValueError("feature number could not be %s" % feature_number)
    return type_list


class HistoryManager:
    def __init__(self, coin_number):
        self._coin_number = coin_number
        self.market_list = ['ChinaA', 'AMEX', 'NYSE', 'NASDAQ', 'CC1', 'CC2']

    def get_global_panel(self, feature_list=['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME'], market='ChinaA'):
        if market not in self.market_list:
            raise ValueError('金融市场数据集不存在')

        if os.path.exists('./database/pkl/' + market + '_' + str(self._coin_number) + '.pkl'):
            panel = pd.read_pickle('./database/pkl/' + market + '_' + str(self._coin_number) + '.pkl')
        else:
            root_path = './database/' + market
            file_list = [f for f in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, f))]

            if self._coin_number > len(file_list):
                raise ValueError('设置提取的资产数量大于可提取范围')

            print('--market: ' + market + '  --coin_number: ' + str(self._coin_number) + 
                '  --feature_number: ' + str(len(feature_list)))

            coin_list = [i[:-4] for i in file_list][:self._coin_number]
            date_list = []
            data_np = []
            
            for i in range(len(file_list)):
                if i == self._coin_number:
                    break
                item_df = pd.read_csv(root_path + '/' + file_list[i])
                item_np = item_df[feature_list].values
                data_np.append(item_np)
                if len(date_list) == 0:
                    date_list = item_df[item_df.columns[0]]

            panel = pd.Panel(items=feature_list, major_axis=coin_list,
                             minor_axis=date_list, dtype=np.float32)

            for i in range(self._coin_number):
                for j in range(len(feature_list)):
                    for k in range(len(date_list)):
                        panel.loc[feature_list[j], coin_list[i], date_list[k]] = data_np[i][k][j]

            print('DataFrame finish!')
            f = open('./database/pkl/' + market + '_' + str(self._coin_number) + '.pkl', 'wb')
            panel.to_pickle(f)
            f.close
        # [features, coins, dates]
        return panel


class DataMatrices:
    def __init__(self, batch_size=50, window_size=20, coin_number=10, feature_number=5, test_portion=0.15,
                 trend_size=20, portion_reversed=False, is_permed=False, buffer_bias_ratio=0, market='ChinaA',
                 picture_bool=False, predict_bool=False):
        self.feature_number = feature_number
        self.__batch_size = batch_size
        self.__window_size = window_size
        self.__trend_size = trend_size
        self.__coin_no = coin_number
        self.test_portion = test_portion
        self.portion_reversed = portion_reversed
        self.__is_permed = is_permed
        self.picture_bool = picture_bool
        self.predict_bool = predict_bool
        self.__features = get_type_list(self.feature_number)
        self.__history_manager = HistoryManager(coin_number=self.__coin_no)
        # [feature, coin, date]
        self.__global_data = self.__history_manager.get_global_panel(feature_list=self.__features, market=market)
        # major: coin, minor: date
        self.__PVM = pd.DataFrame(np.ones((len(list(self.__global_data.minor_axis)), self.__coin_no))/self.__coin_no,
                                 index=list(self.__global_data.minor_axis),
                                 columns=list(self.__global_data.major_axis))
        self.__num_periods = len(self.__global_data.minor_axis)
        self.__divide_data(test_portion, portion_reversed)

        self.__replay_buffer = ReplayBuffer(start_index=self._train_ind[0],
                                            end_index=self._train_ind[-1],
                                            sample_bias=buffer_bias_ratio,
                                            batch_size=self.__batch_size,
                                            coin_number=self.__coin_no,
                                            is_permed=self.__is_permed)

        print("the number of training examples is %s"
              ", of test examples is %s" % (self._num_train_samples, self._num_test_samples))
        print("the training set is from %s to %s" %
              (min(self._train_ind), max(self._train_ind)))
        print("the test set is from %s to %s" %
              (min(self._test_ind), max(self._test_ind)))

    def get_test_set(self):
        return self.__pack_samples(self._test_ind)
    
    def get_test_set_online(self):
        return self.__pack_samples_test_online(self._test_ind[0], self._test_ind[-1], self.__window_size)

    def next_batch(self):
        batch = self.__pack_samples(
            [exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        return batch

    def __pack_samples(self, indexs):
        indexs = np.array(indexs)
        last_w = self.__PVM.values[indexs-1, :]
        def setw(w):
            self.__PVM.iloc[indexs, :] = w

        indexs = np.array(indexs)
        # M: [batch, feature, coin, seq]
        # feature: 'CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME'
        M = [self.get_submatrix(index) for index in indexs]
        M = np.array(M)


        data_ps = M[:, :-1, :, self.__window_size:self.__window_size + self.__window_size]
        data_ps = data_ps / data_ps[:, 0:1, :, -1:]

        relative_price = M[:, :-1, :, self.__window_size+self.__window_size] / M[:, 0, None, :, self.__window_size+self.__window_size-1] 

        # 序列和图片数据做趋势预测
        data_cls = data_ps / data_ps[:, 0:1, :, -1:]

        if self.picture_bool:
            price_picture = self.get_price_picture(M)
        else:
            price_picture = None
        
        if self.predict_bool:
            predict_label = predict(data_cls, './checkpoint_cls/lstm.pkl')
        else:
            predict_label = None

        # trend [batch, coin]
        trend = M[:, 0, :, -1] / M[:, 0, :, -(self.__trend_size + 1)]
        trend = trend > 1
        trend = trend + 0

        return {"data_ps": data_ps, "relative_price": relative_price, "last_w": last_w, "setw": setw,
                "data_cls": data_cls, "price_picture": price_picture, "trend": trend, "predict_label": predict_label}
    
    def __pack_samples_test_online(self, ind_start, ind_end, x_window_size):
        last_w = self.__PVM.values[ind_start-1:ind_start, :]

        def setw(w):
            self.__PVM.iloc[ind_start, :] = w
        M = [self.get_submatrix_test_online(
            ind_start, ind_end)]  # [1,4,11,2807]
        M = np.array(M)
        data_ps = M[:, :-1, :, :-self.__trend_size]
        data_ps = data_ps / data_ps[:, 0:1, :, -1:]
        if self.__trend_size==1:
            relative_price = M[:, :-1, :, x_window_size:] / M[:, 0, None, :, x_window_size-1:-1]
        else:
            relative_price = M[:, :-1, :, x_window_size:-(self.__trend_size-1)] / M[:, 0, None, :, x_window_size-1:-self.__trend_size]
        return {"data_ps": data_ps, "relative_price": relative_price, "last_w": last_w, "setw": setw}

    def get_submatrix(self, ind):
        # [feature, coin, date]
        return self.__global_data.values[:, :, ind - self.__window_size:ind+self.__window_size + self.__trend_size]
    
    def get_submatrix_test_online(self, ind_start, ind_end):
        return self.__global_data.values[:, :, ind_start:ind_end]

    def __divide_data(self, test_portion, portion_reversed):
        train_portion = 1 - test_portion
        s = float(train_portion + test_portion)
        if portion_reversed:
            portions = np.array([test_portion]) / s
            portion_split = (portions * self.__num_periods).astype(int)
            indices = np.arange(self.__num_periods)
            self._test_ind, self._train_ind = np.split(indices, portion_split)
        else:
            portions = np.array([train_portion]) / s
            portion_split = (portions * self.__num_periods).astype(int)
            indices = np.arange(self.__num_periods)
            self._train_ind, self._test_ind = np.split(indices, portion_split)

        self._train_ind = self._train_ind[self.__window_size:-(self.__window_size + self.__trend_size)]
        self._train_ind = list(self._train_ind)
        self._test_ind = self._test_ind[:-(self.__window_size + self.__trend_size)]
        self._test_ind = list(self._test_ind)
        self._num_train_samples = len(self._train_ind)
        self._num_test_samples = len(self._test_ind)

    def get_fixlen_iter(self):
        rest = self._num_train_samples % self.__batch_size
        start = rest
        for i in range(start, self._num_train_samples, self.__batch_size):
            batch = self._train_ind[i:i + self.__batch_size]
            yield self.__pack_samples(batch)

    def __iter__(self):
        return self.get_fixlen_iter()
    
    def get_price_picture(self, M):
        x = M[:, :-1, :, self.__window_size:-self.__trend_size]
        volume = M[:, -1:, :, self.__window_size:-self.__trend_size]

        MA = [np.mean(M[:, :1, :, a:a+self.__window_size], axis=3) for a in range(self.__window_size)]
        MA = np.array(MA)
        MA = MA.transpose((1, 2, 3, 0))

        x = np.concatenate((x, MA), axis=1)
        x = np.concatenate((x, volume), axis=1)
        x = x.transpose((0, 2, 3, 1))

        data = []

        for i in range(x.shape[0]):
            item = []
            for j in range(x.shape[1]):
                data_item = x[i][j]
                image = get_image_with_price(data_item)
                tensor_value = image_loader(image)
                list_value = tensor_value.numpy().tolist()
                item.append(list_value)
            data.append(item)
        data = torch.Tensor(data)
        return data


if __name__ == '__main__':
    batch_size = 32
    window_size = 30
    trend_size = 20
    coin_number = 36
    feature_number = 5
    market='CC2'
    DM = DataMatrices(batch_size=batch_size, window_size=window_size, coin_number=coin_number, feature_number=feature_number,
                      test_portion=0.15,trend_size=trend_size, portion_reversed=False, is_permed=True,
                      buffer_bias_ratio=5e-5, market=market, picture_bool=False, predict_bool=False)

    nextbatch = DM.next_batch()
    data_ps = nextbatch['data_ps']
    relative_price = nextbatch['relative_price']
    last_w = nextbatch['last_w']
    data_cls = nextbatch['data_cls']
    price_picture = nextbatch['price_picture']
    trend = nextbatch['trend']
    predict_label = nextbatch['predict_label']
    print(data_ps.shape)
    print(relative_price.shape)
    print(last_w.shape)
    print(data_cls.shape)
    #print(price_picture.shape)
    print(trend.shape)
    #print(predict_label.shape)
    testset = DM.get_test_set_online()
    dp = testset['data_ps']
    rp = testset['relative_price']
    lw = testset['last_w']
    print(dp.shape)
    print(rp.shape)
    print(lw.shape)
            

















