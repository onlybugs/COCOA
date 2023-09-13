import numpy as np
# import pandas as pd
from MFGenhffh import MFGen
import torch as t
from tqdm import tqdm

epi_read  = "mean"
read_type = "hff-block-per-mf" 
model = t.load("./have/RegModule-Best.pt",map_location = t.device('cpu')) 

def calcu(lepi:t.Tensor,repi:t.Tensor) -> t.Tensor:
    """"
    Calculate l epi signal and r epi signal to a predicted-corr block.

    Parameters
    ----------
    lepi: left epi signal.
    repi: right epi signal.

    Returns
    -------
    Predicted result.

    Examples
    --------
    None

    """
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
    '''
    Predict corr blocks and combined them to a corr mat. 

    Parameters
    ----------
    epic_mat: x * y. Row is a kind of epic. Col reprents genome region.
    bsize: The size of blocks.

    Returns
    -------
    Combind corr.

    Examples
    --------
    None

    '''
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

filelist = ["H3K27ac","H3K4me1","H3K4me3","H3K9me3","H3K27me3","H3K36me3"]
per_dict = {"max":100,"mean":99,"median":99}
chrs_list = ['12' ,'13' ,'14' ,'15' ,'16' ,'17' ,'18' ,'19'] # "14",
epi_dir_list = ['gm12878']# ["activated_B","activated_T","amnio","chorio","NCI_H929","SJSA1"]

save_path = "cell_type_predict/"

for epi_name in epi_dir_list:

    for chr in chrs_list:
        epic_l = []
        for f in filelist:
            seq = np.load("data/epi/"+ epi_name +"/"+f+"_"+epi_read+".npz")['chr'+chr]
            seq = seq - np.nanmin(seq)
            seq[np.isnan(seq)] = 0

            seq = np.log1p(seq)
            seq = minmax(seq)

            epic_l.append(seq) # (seq[:20000])


        epic_mat = np.array(epic_l)
        print("Begin combine:")
        pred_y = together(epic_mat,bsize=128)

        np.savez(save_path + epi_name + "-" + chr +".npz",pre_out = pred_y)
        print("Chr " + chr + " on " + epi_name + " is finish! ^_^")

