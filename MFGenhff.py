import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

import numpy as np

import math

class ResLayer2D(nn.Module):
    def __init__(self,in_channel,out_channel,ks = 1,p = 0,downsample = False,s = 1):
        super(ResLayer2D,self).__init__()
        self.feature = nn.Sequential(OrderedDict([
            ("conv1",nn.Conv2d(in_channel,out_channel,kernel_size=ks,padding=p,bias=False,stride=s)),
            ("norm1",nn.BatchNorm2d(out_channel)),
            ("relu1",nn.Tanh()),
            ("conv2",nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1,bias=False,stride=1)),
            ("norm2",nn.BatchNorm2d(out_channel))
        ]))

        self.downsample = downsample
        if(downsample):
            self.ds = nn.Conv2d(in_channel,out_channel,1)

        self.relu = nn.Tanh()

    def forward(self,x):
        residual = x
        new_features = self.feature(x)
        if self.downsample:
            residual = self.ds(residual)
            
        new_features += residual

        # relu
        return self.relu(new_features)

class ResBlock2D(nn.Module):
    def __init__(self,num_layers,in_channel,out_channel,ks = 1,p = 0,s = 1):
        super(ResBlock2D,self).__init__()
        simple_block = []
        for i in range(num_layers):
            if(i == 0):
                self.upfeature = ResLayer2D(in_channel,out_channel,ks,p,True,s)
            else:
                simple_block.append(('res'+str(i+1),ResLayer2D(out_channel,out_channel,3,1,False,s)))
                
        self.downfeature = nn.Sequential(OrderedDict(simple_block))
        # self.pool = nn.AdaptiveAvgPool2d(psize)

    def forward(self,x):

        x = self.upfeature(x)
        x = self.downfeature(x)

        return x


class MF(nn.Module):
    def __init__(self,in_channel,out_channel,N) -> None:
        super(MF,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,(N,1),bias=False)
        self.relu = nn.Tanh()
        self.delin = nn.Parameter(torch.randn(out_channel,1,1))
        self.relin = nn.Parameter(torch.randn(out_channel,1,1))

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = x.mul(self.delin)
        xt = x.permute([0,1,3,2])
        x = (x * xt).mul(self.relin)

        return x

class AFF(nn.Module):
    '''
    AFF
    '''
    def __init__(self, channels=16, r=8,up = False):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.up = up

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        self.upconv = nn.Conv2d(r,channels,1,bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        if(self.up):
            residual = self.upconv(residual)
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class MFGen(nn.Module):
    def __init__(self) -> None:
        super(MFGen,self).__init__()
        
        # 1D
        self.mfl = MF(1,128,6)
        self.mfr = MF(1,128,6)

        self.afflr = AFF(128,8)
        self.affrl = AFF(128,8)

        # 2D
        self.conv2D = nn.Sequential(OrderedDict([
            ("2dconv0",ResBlock2D(6,256,128)),
            ("2dconv1",ResBlock2D(6,128,64)),
            ("2dconv2",ResBlock2D(6,64,32)),
            ("2dconv3",ResBlock2D(6,32,16)),
            ("2dconv4",ResBlock2D(6,16,8)),
            ("2dconv5",ResBlock2D(6,8,1))
        ]))


    def forward(self,xl,xr):
        # 1d mf
        xl = self.mfl(xl)
        xr = self.mfr(xr)

        # aff
        xlr = self.afflr(xl,xr)
        xrl = self.affrl(xr,xl)
        xres = torch.concat([xlr,xrl],axis = 1)

        # 2d res
        xcc = self.conv2D(xres)
        # xcc = xcc.squeeze()


        return xcc

