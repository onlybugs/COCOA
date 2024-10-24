# Predict

import numpy as np
from MFGenhffh import MFGen
from model import COCOA
import torch as t
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mp',help="Pre-trained Model Path",type=str)
parser.add_argument('-i',help="Input data dir (*.npz)",type=str)
parser.add_argument('-o',help="Output data dir",type=str)
parser.add_argument('-c',help="Chr number",nargs="+")

args = parser.parse_args()
##############################################################
model_path = args.mp # example as : ./pretrained/RegModule-Best.pt
inp_dir = args.i # example as : ../data/epi/hff1/
save_path = args.o
chrs_list = args.c
#############################################################


def calcu(lepi:t.Tensor,repi:t.Tensor) -> t.Tensor:

    # sub_epi[t.isnan(sub_epi)] = 0
    l = lepi.float().unsqueeze(0).unsqueeze(1)
    r = repi.float().unsqueeze(0).unsqueeze(1)
    with t.no_grad():
        y = model(l,r)
    y = y.squeeze()

    return y

def minmax(mat):
    return (mat - np.min(mat)) / (np.max(mat) - np.min(mat))

# Together
def together(epic_mat:np.array,bsize:int = 128) -> np.array:

    jump = bsize // 2
    lens = epic_mat.shape[1]
    out = np.zeros((lens,lens))

    combar = tqdm(range(0,lens-jump+1,jump))
    for i in combar:
        if(lens - i < bsize): i = lens - bsize
        for j in range(i,lens-jump+1,jump):
                if(lens - j < bsize): j = lens - bsize
                l = epic_mat[:,i:i+bsize]
                r = epic_mat[:,j:j+bsize]
                # epi = np.concatenate([l,r],axis=1)

                o = calcu(t.from_numpy(l),t.from_numpy(r))
                out[i:i+bsize,j:j+bsize] = o.numpy()
                combar.set_description(desc=f"[{i,j}/{lens}]")

    return np.triu(out) + np.tril(out.T,-1)

model = t.load(model_path,map_location = t.device('cpu'))
filelist = ["H3K27ac","H3K4me1","H3K4me3","H3K9me3","H3K27me3","H3K36me3"]

for chr in chrs_list:
    epic_l = []
    for f in filelist:
        seq = np.load(inp_dir+f+".npz")['chr'+chr]
        seq = seq - np.nanmin(seq)
        seq[np.isnan(seq)] = 0
        seq = np.log1p(seq)
        seq = minmax(seq)
        epic_l.append(seq)

    epic_mat = np.array(epic_l)
    print("Begin combine:")
    pred_y = together(epic_mat,bsize=128)
    np.savez(save_path + chr +"-pre.npz",pre_out = pred_y)
    print("Chr " + chr + " prediction is finish! ^_^")
