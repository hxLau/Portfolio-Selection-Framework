import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def add_noise(signal):
    SNR = 80
    for i in range(signal.shape[2]):
      noise = np.random.randn(signal.shape[0], signal.shape[1], 1, signal.shape[3])
      noise = noise-np.mean(noise)
      signal_p = signal[:,:,i:i+1,:]
      signal_power = np.linalg.norm( signal_p )**2 / signal_p.size
      noise_variance = signal_power/np.power(10,(SNR/10)) 
      noise = (np.sqrt(noise_variance) / np.std(noise) )*noise 
      signal[:,:,i:i+1,:] += noise
    return signal

def add_noise_encoder(signal):
    signal = signal.detach().cpu().numpy()
    SNR = 80
    noise = np.random.randn(signal.shape[0], signal.shape[1], signal.shape[2], signal.shape[3])
    noise = noise-np.mean(noise)
    signal_power = np.linalg.norm( noise )**2 / noise.size
    noise_variance = signal_power/np.power(10,(SNR/10)) 
    noise = (np.sqrt(noise_variance) / np.std(noise) )*noise 
    signal += noise
    return torch.tensor(signal, dtype=torch.float).cuda()

def add_noise_decoder(signal):
    signal = signal.detach().cpu().numpy()
    SNR = 80
    noise = np.random.randn(signal.shape[0], signal.shape[1], signal.shape[2])
    noise = noise-np.mean(noise)
    signal_power = np.linalg.norm( noise )**2 / noise.size
    noise_variance = signal_power/np.power(10,(SNR/10)) 
    noise = (np.sqrt(noise_variance) / np.std(noise) )*noise 
    signal += noise
    return torch.tensor(signal, dtype=torch.float).cuda()