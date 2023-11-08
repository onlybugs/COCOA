import torch as t
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset,DataLoader

import numpy as np
from tqdm import tqdm
import sys
from math import log10

from MFGenhffh import MFGen
from model import COCOA

from utils.SSIM import ssim
from torchvision.models.vgg import vgg16

# init 3e-4
# sec 5e-4
class Config():
    trainfp = "train.npz"
    testfp = "val.npz"
    logn = "log.txt"
    msave_path = "./msave/"
    batch_size = 16
    lr = 5e-4
    decay = 0.8
    epoc_num = 640
    decay_epoc = 64
    best_ssim = -2

def wlog(fname,w):

    with open(fname,'a+') as f:
        f.write(w)
        f.close()

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        b, c, h, w = x.shape
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = t.pow((x[:, :, 1:, :] - x[:, :, :h-1, :]), 2).sum()
        w_tv = t.pow((x[:, :, :, 1:] - x[:, :, :, :w-1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b

    @staticmethod
    def tensor_size(t1):
        return t1.size()[1] * t1.size()[2] * t1.size()[3]

cfg = Config()
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
cTrain = sys.argv[1]
# DataSet
class Mydata(Dataset):
    def __init__(self,fpath) -> None:
        super().__init__()
        rdata = np.load(fpath)
        self.xl = rdata['xl']
        self.xr = rdata['xr']
        self.y = rdata['y']

    def __len__(self):
        
        return self.xl.shape[0]

    def __getitem__(self, index):
        
        return self.xl[index],self.xr[index],self.y[index]

tdata = Mydata(cfg.trainfp)
testdata = Mydata(cfg.testfp)
test_loader = DataLoader(testdata,shuffle=True,batch_size=cfg.batch_size,drop_last = True)

if(cTrain == "0"):
    mdl = MFGen().to(device)
    print("We train model from 0.")
elif(cTrain == '1'):
    print("We continue train model...\n")
    mdl = t.load("msave/RegModule-Best.pt")


creMSE = nn.MSELoss(size_average = False)
creMAE = nn.L1Loss(size_average = False)
tvloss = TVLoss()
vgg = vgg16(pretrained=True)
loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
for param in loss_network.parameters():
    param.requires_grad = False
loss_network = loss_network.to(device)
optimizer = optim.Adam(mdl.parameters(),lr = cfg.lr)
lrdecay = optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=cfg.decay_epoc, gamma=cfg.decay)


for e in range(cfg.epoc_num):
    train_loader = DataLoader(tdata,shuffle=True,batch_size=cfg.batch_size,drop_last = True)
    mdl.train()
    train_bar = tqdm(train_loader)
    tloss = 0
    for i,(xl,xr,y) in enumerate(train_bar):
        # print(x.shape)
        xl = xl.unsqueeze(1).float().to(device)
        xr = xr.unsqueeze(1).float().to(device)
        y = y.unsqueeze(1).float().to(device) # bs 64 64
        fake = mdl(xl,xr)
        # print(fake)

        mseloss = creMSE(fake,y)
        maeloss = creMAE(fake,y)
        out_feat = loss_network(fake.repeat([1,3,1,1]))
        target_feat = loss_network(y.repeat([1,3,1,1]))
        perception_loss = creMSE(out_feat.reshape(out_feat.size(0),-1), \
                                target_feat.reshape(target_feat.size(0),-1))
        loss = mseloss + perception_loss * 0.002 + tvloss(fake.cpu()).to(device) * 2e-8

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tloss += loss
        
        train_bar.set_description(\
                desc=f"[{e}/{cfg.epoc_num}] Loss: {loss.cpu().detach().numpy():.8f}")
        # output
        if (i+1) % 100 == 0:
            l = "train: epoch %d,batch %d,100 batch avg loss %.8f.\n" % (e,i,tloss.cpu().detach().numpy() / 100)
            wlog(cfg.logn,l)
            tloss = 0

    # eval
    mdl.eval()
    valid_bar = tqdm(test_loader)
    ssim_all = 0
    with t.no_grad():
        for j,(xl,xr,y) in enumerate(valid_bar):
            xl = xl.unsqueeze(1).float().to(device)
            xr = xr.unsqueeze(1).float().to(device)
            y = y.unsqueeze(1).float().to(device) # bs 64 64
            fake = mdl(xl,xr)

            batch_mse = ((fake-y) ** 2).mean()
            batch_ssim = ssim(fake, y)
            psnr = 10 * log10(1 / batch_mse)
            # batch_mse = ((fake - y) ** 2).mean()
            batch_mae = (abs(fake - y)).mean()

            ssim_all += batch_ssim

            l = "val: epoch %d,batch %d,ssim %.8f,psnr %.8f.\n" % (e,i,batch_ssim,psnr)
            wlog(cfg.logn,l)
            valid_bar.set_description(desc=f"[Predicting in Test set]\
               ssim: {batch_ssim:.6f} psnr: {psnr:.6f}")

    # Best
    ssim_all /= j
    if ssim_all > cfg.best_ssim:
        cfg.best_ssim = ssim_all
        l = "Get best ssim %.8f from epoch %d batch %d\n" % (cfg.best_ssim,e,i)
        print(l)
        wlog(cfg.logn,l)
        t.save(mdl,cfg.msave_path + "RegModule-Best.pt")
    if e % 5 == 0:
        t.save(mdl,cfg.msave_path + "RegModule"+ "-" + str(e)+".pt")

    lrdecay.step()
