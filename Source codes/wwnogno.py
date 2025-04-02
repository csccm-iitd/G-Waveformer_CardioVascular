from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from integral_transform import IntegralTransform
from neighbor_search import NeighborSearch
from wno_block import *
device = torch.device('cpu')
# device= torch.device('cuda:1')
class WNOGNO(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            in_channels2,
            out_channels2,
            in_channels3,
            projection_channels=512,
            embed_dim= 25,
            wavelet = 'db6',
            level = 3,        
            width = 9,       
            layers = 2,  
            gno_coord_dim=3,
            grid_range1 = [1, 1, 1],
            gno_coord_embed_dim=None,
            gno_mlp_hidden_layers=[512, 256],
            gno_mlp_non_linearity=F.gelu, 
            gno_transform_type=2,
            gno_use_open3d=True,
            **kwargs
        ):
        
        
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels2 = in_channels2
        self.out_channels2 = out_channels2
        self.embed_dim = embed_dim
        # self.embed_dim1 = embed_dim1
        self.gno_encode = IntegralTransform(mlp_layers=[in_channels,gno_mlp_hidden_layers[0], gno_mlp_hidden_layers[1],out_channels],transform_type=2)   
        # self.FNO_1 = FNO3d(embed_dim,embed_dim,embed_dim,n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
        # wavelet = 'db6'  
        # level = 2        
        # width = 12       
        # layers = 2      
        # sub = 1         
        # h = embed_dim           
        # in_channel2 = 1
        # grid_range1 = [1, 1, 1]

        self.WNO_1 = WNO2d(width, level, layers=layers, size=[self.embed_dim, self.embed_dim, self.embed_dim], wavelet=wavelet,
                        in_chanel=in_channels2, grid_range=grid_range1).to(device)
        
        self.gno_decode = IntegralTransform(mlp_layers=[in_channels3,gno_mlp_hidden_layers[0],gno_mlp_hidden_layers[1],out_channels2],transform_type=2)
        
        # gno = IntegralTransform(mlp_layers=[in_channels,gno_mlp_hidden_layers[0],gno_mlp_hidden_layers[1],out_channels],transform_type=2)
        
        # FNO_1 = FNO3d(embed_dim,embed_dim,embed_dim,n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
        
        # gno2 = IntegralTransform(mlp_layers=[in_channels,gno_mlp_hidden_layers[0],gno_mlp_hidden_layers[1],out_channels],transform_type=2)

    def forward(self,f_y1,x_cord,x_cord1,nbrs,nbrs2):
        
        #Compute latent space embedding
        # f_y11 = f_y1[:,:-1]
        # f_y12 = f_y1[:,1:]
        
        x_cord_in = x_cord
        # f_y11 = f_y11
        # f_y12 = f_y12
        y_dash = torch.cat([x_cord_in,f_y1],1)
        # y_dash1 = torch.cat([x_cord_in,f_y11],1)
        # y_dash2 = torch.cat([x_cord_in,f_y12],1)
        
        # gno_out1  = self.gno_encode(y=y_dash1,neighbors=nbrs,x=x_cord1)
        # gno_out2  = self.gno_encode(y=y_dash2,neighbors=nbrs,x=x_cord1)
        
        gno_out = self.gno_encode(y=y_dash,neighbors=nbrs,x=x_cord1)
        gno_out = gno_out.reshape(1,self.embed_dim,self.embed_dim,self.embed_dim,f_y1.shape[-1]).to(device)
        gno_out1  = gno_out[:,:,:,:,:-1]
        gno_out2  =  gno_out[:,:,:,:,1:]
        
        # gno_out1 = gno_out1.reshape(1,self.embed_dim,self.embed_dim,self.embed_dim,1).to(device)
        # gno_out2 = gno_out2.reshape(1,self.embed_dim,self.embed_dim,self.embed_dim,1).to(device)
        
        #Fno 
        out_f = self.WNO_1(gno_out1,gno_out2).cpu()
        out2 =  out_f.reshape(self.embed_dim*self.embed_dim*self.embed_dim,1)
        y_dash2 = torch.cat([x_cord1,out2],1)
        # y_dash2 = out2
        
        #Integrate latent space
        out = self.gno_decode(y=y_dash2,neighbors=nbrs2,x=x_cord_in)
        return out
