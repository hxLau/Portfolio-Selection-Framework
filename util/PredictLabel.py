import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_cls.LSTMTagger import LSTMTagger


def predict(X, model_path):
    # batch, feature, asset, window
    model = torch.load(model_path)
    X = torch.tensor(X, dtype=torch.float).cuda()
    window_size = X.shape[3]
    batch_size = X.shape[0]
    feature_size = X.shape[1]
    asset_size = X.shape[2]
    X = X.permute((3, 0, 2, 1)).contiguous()
    X = X.view(window_size, batch_size*asset_size, feature_size)

    tags = model(X)
    prediction = tags.argmax(dim = 1)
    prediction = prediction.view(batch_size, asset_size, 1)
    return prediction