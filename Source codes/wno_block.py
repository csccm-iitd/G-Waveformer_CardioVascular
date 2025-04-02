# from IPython import get_ipython
# get_ipython().magic('reset -sf')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

from timeit import default_timer
from utils import *
# from utilites3 import *
import ptwt, pywt
from ptwt.conv_transform_3 import wavedec3, waverec3

from einops import rearrange
from einops.layers.torch import Rearrange
import module
from module import ConvAttention1, ConvAttention2, PreNorm, PreNorm2, FeedForward
from torch.optim import lr_scheduler as lr_scheduler
from trgcvt_module import Transformer1, Transformer2, CvTencoderdecoder1, CvTencoderdecoder2

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cpu')
# device= torch.device('cuda:1')
# %%
""" Def: 2d Wavelet layer """
class WaveConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet):
        super(WaveConv3d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.wavelet = wavelet
        dummy_data = torch.randn( [*size] ).unsqueeze(0)
        mode_data = wavedec3(dummy_data, pywt.Wavelet(self.wavelet),
                             level=self.level, mode='periodic')
        self.modes1 = mode_data[0].shape[-3]
        self.modes2 = mode_data[0].shape[-2]
        self.modes3 = mode_data[0].shape[-1]

        self.scale = (1 / (in_channels * out_channels))
        self.transformer_1 = CvTencoderdecoder1(self.modes1,self.in_channels,self.in_channels,self.out_channels)
        self.transformer_2 = CvTencoderdecoder1(self.modes1,self.in_channels,self.in_channels,self.out_channels)
        self.transformer_3 = CvTencoderdecoder1(self.modes1,self.in_channels,self.in_channels,self.out_channels)
        self.transformer_4 = CvTencoderdecoder1(self.modes1,self.in_channels,self.in_channels,self.out_channels)
        self.transformer_5 = CvTencoderdecoder1(self.modes1,self.in_channels,self.in_channels,self.out_channels)
        self.transformer_6 = CvTencoderdecoder1(self.modes1,self.in_channels,self.in_channels,self.out_channels)
        self.transformer_7 = CvTencoderdecoder1(self.modes1,self.in_channels,self.in_channels,self.out_channels)
        self.transformer_8 = CvTencoderdecoder1(self.modes1,self.in_channels,self.in_channels,self.out_channels)

    # Convolution
    def mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        # return torch.einsum("bixyz,ioxyz->boxyz", input, weights)
        return torch.einsum("ixyz,ioxyz->oxyz", input, weights)

    def forward(self, x_1,x_2):
        xr = torch.zeros(x_1.shape, device = x_1.device)
        for i in range(x_1.shape[0]):
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            x_coeff1 = wavedec3(x_1[i, ...], pywt.Wavelet(self.wavelet), level=self.level, mode='periodic')
            x_coeff2 = wavedec3(x_2[i, ...], pywt.Wavelet(self.wavelet), level=self.level, mode='periodic')
            
            # Multiply relevant Wavelet modes
            # x_coeff1[0] = self.mul2d(x_coeff[0].clone(), self.weights1)
            # x_coeff1[1]['aad'] = self.mul2d(x_coeff[1]['aad'].clone(), self.weights2)
            # x_coeff1[1]['ada'] = self.mul2d(x_coeff[1]['ada'].clone(), self.weights3)
            # x_coeff1[1]['add'] = self.mul2d(x_coeff[1]['add'].clone(), self.weights4)
            # x_coeff1[1]['daa'] = self.mul2d(x_coeff[1]['daa'].clone(), self.weights5)
            # x_coeff1[1]['dad'] = self.mul2d(x_coeff[1]['dad'].clone(), self.weights6)
            # x_coeff1[1]['dda'] = self.mul2d(x_coeff[1]['dda'].clone(), self.weights7)
            # x_coeff1[1]['ddd'] = self.mul2d(x_coeff[1]['ddd'].clone(), self.weights8)
            # Multiply relevant Wavelet modes
            x_coeff1[0] = self.transformer_1(x_coeff1[0].unsqueeze(0).clone(),x_coeff2[0].unsqueeze(0).clone()).squeeze(0)
            x_coeff1[1]['aad'] = self.transformer_2(x_coeff1[1]['aad'].unsqueeze(0).clone(), x_coeff2[1]['aad'].unsqueeze(0).clone()).squeeze(0)
            x_coeff1[1]['ada'] = self.transformer_3(x_coeff1[1]['ada'].unsqueeze(0).clone(), x_coeff2[1]['ada'].unsqueeze(0).clone()).squeeze(0)
            x_coeff1[1]['add'] = self.transformer_4(x_coeff1[1]['add'].unsqueeze(0).clone(), x_coeff2[1]['add'].unsqueeze(0).clone()).squeeze(0)
            x_coeff1[1]['daa'] = self.transformer_5(x_coeff1[1]['daa'].unsqueeze(0).clone(), x_coeff2[1]['daa'].unsqueeze(0).clone()).squeeze(0)
            x_coeff1[1]['dad'] = self.transformer_6(x_coeff1[1]['dad'].unsqueeze(0).clone(), x_coeff2[1]['dad'].unsqueeze(0).clone()).squeeze(0)
            x_coeff1[1]['dda'] = self.transformer_7(x_coeff1[1]['dda'].unsqueeze(0).clone(), x_coeff2[1]['dda'].unsqueeze(0).clone()).squeeze(0)
            x_coeff1[1]['ddd'] = self.transformer_8(x_coeff1[1]['ddd'].unsqueeze(0).clone(), x_coeff2[1]['ddd'].unsqueeze(0).clone()).squeeze(0)
            
            # x_coeff1[0] = self.transformer_1(x_coeff1[0].unsqueeze(0).clone(),x_coeff2[0].unsqueeze(0).clone()).squeeze(0)
            # x_coeff1[1]['aad'] = self.transformer_2(x_coeff1[1]['aad'].unsqueeze(0).clone(), x_coeff2[1]['aad'].unsqueeze(0).clone()).squeeze(0)
            # x_coeff1[1]['ada'] = self.transformer_2(x_coeff1[1]['ada'].unsqueeze(0).clone(), x_coeff2[1]['ada'].unsqueeze(0).clone()).squeeze(0)
            # x_coeff1[1]['add'] = self.transformer_2(x_coeff1[1]['add'].unsqueeze(0).clone(), x_coeff2[1]['add'].unsqueeze(0).clone()).squeeze(0)
            # x_coeff1[1]['daa'] = self.transformer_2(x_coeff1[1]['daa'].unsqueeze(0).clone(), x_coeff2[1]['daa'].unsqueeze(0).clone()).squeeze(0)
            # x_coeff1[1]['dad'] = self.transformer_2(x_coeff1[1]['dad'].unsqueeze(0).clone(), x_coeff2[1]['dad'].unsqueeze(0).clone()).squeeze(0)
            # x_coeff1[1]['dda'] = self.transformer_2(x_coeff1[1]['dda'].unsqueeze(0).clone(), x_coeff2[1]['dda'].unsqueeze(0).clone()).squeeze(0)
            # x_coeff1[1]['ddd'] = self.transformer_2(x_coeff1[1]['ddd'].unsqueeze(0).clone(), x_coeff2[1]['ddd'].unsqueeze(0).clone()).squeeze(0)
            
            
            # Instantiate higher level coefficients as zeros
            for jj in range(2, self.level + 1):
                x_coeff1[jj] = {key: torch.zeros([*x_coeff1[jj][key].shape], device=device)
                                for key in x_coeff1[jj].keys()}
            
            # Return to physical space        
            xr[i, ...] = waverec3(x_coeff1, pywt.Wavelet(self.wavelet))
        return xr

""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_chanel, grid_range):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.level = level
        self.width = width
        self.size = size
        self.inp_size = in_chanel
        # print(self.inp_size)
        self.layers = layers
        self.grid_range = grid_range 
        self.padding = 1 # pad the domain if input is non-periodic
                
        # self.conv = nn.ModuleList()
        # self.w = nn.ModuleList()
        # print(self.size,self.width,self.width,self.width)
        
        self.fc0 = nn.Linear(self.inp_size, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = WaveConv3d(self.width, self.width, self.level, self.size, wavelet)
        self.w0 =  CvTencoderdecoder2(self.size[0],self.width,self.width,self.width)
        
        # for i in range(self.layers):
        #     self.conv.append(WaveConv3d(self.width, self.width, self.level, 
        #                                 self.size, wavelet))
        #     self.w.append(nn.Conv3d(self.width, self.width, 1))
            
        
        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, y):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)                 # Shape: Batch * x * y * z * Channel
        y = self.fc0(y) 
        x = x.permute(0, 4, 3, 1, 2)    # Shape: Batch * Channel * z * x * y 
        y = y.permute(0, 4, 3, 1, 2)
        
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding, 0,self.padding]) # do padding, if required
            
        if self.padding != 0:
            y = F.pad(y, [0,self.padding, 0,self.padding, 0,self.padding]) # do padding, if required
            
        
        x = self.conv0(x,y) + self.w0(x,y) 
        # for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
        #     x = convl(x) + wl(x) 
        #     if index != self.layers - 1:     # Final layer has no activation    
        #         x = F.mish(x)                # Shape: Batch * Channel * x * y
        
        
            
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding, :-self.padding] # remove padding, when required
        x = x.permute(0, 3, 4, 2, 1)        # Shape: Batch * x * y * z * Channel 
        x = self.fc2(F.mish(self.fc1(x)))   # Shape: Batch * x * y * z 
        return x
    
    # def get_grid(self, shape, device):
    #     batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
    #     gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    #     gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
    #     gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    #     gridz = torch.tensor(np.linspace(0, self.grid_range[2], size_z), dtype=torch.float)
    #     gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    #     return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


# batch_size = 10
# learning_rate = 0.001

# epochs = 500
# step_size = 50   # weight-decay step size
# gamma = 0.5      # weight-decay rate

# wavelet = 'db6'  # wavelet basis function
# level = 2        # lavel of wavelet decomposition
# width = 40       # uplifting dimension
# layers = 4       # no of wavelet layers

# sub = 1          # subsampling rate
# h = 64           # total grid size divided by the subsampling rate
# grid_range = [1, 1, 1]
# in_channel = 13  # input channel is 12: (10 for a(x,t1-t10), 2 for x)

# T_in = 10
# T = 20           # No of prediction steps
# step = 1         # Look-ahead step size



# # %%
# """ The model definition """
# model = WNO2d(width, level, layers=layers, size=[T, h, h], wavelet=wavelet,
#               in_chanel=in_channel, grid_range=grid_range).to(device)
# print(count_params(model))

