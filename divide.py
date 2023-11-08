import numpy as np
import pandas as pd
import cooler

##########################################################
micro_path = "/path/to/micro-c/data"
epi_path = "---" # example as : ../data/epi/hff1/
chrs_list = ['5' ,'6' ,'7' ,'8' ,'9' ]
store_dir = "---"
########################################################

def divideblock(micro_mat,epic_mat,bsize = 128):
    '''
    epic_mat: x * y. Row is a kind of epic. Col reprents genome region.
    micro_mat: y * y. same size to epic y.
    '''
    jump = 64
    el = []
    er = []
    h = []
    chrlen = micro_mat.shape[0]
    for i in range(0,chrlen-bsize+1,jump):
        for j in range(i,chrlen-bsize+1,jump):
            h.append(micro_mat[i:i+bsize,j:j+bsize])
            el.append(epic_mat[:,i:i+bsize])
            er.append(epic_mat[:,j:j+bsize])

    return np.array(el),np.array(er),np.array(h)

def GetOE(mat):
    '''
    Get OE
    '''
    # mat = self.rmat
    chr_len = mat.shape[0]
    cut_off = chr_len/100
    mask = np.zeros(chr_len)
    num_mat = mat.copy()
    num_mat[num_mat > 0] = 1
    num_vector = np.sum(num_mat,axis=0)
    for i in range(chr_len):
        if(num_vector[i] >= cut_off):
            mask[i] = 1
    mask = mask == 1

    ox = np.arange(chr_len)
    oy = np.arange(chr_len)
    omask = mask.copy()
    decay = {}
    for i in range(chr_len):
        o_diag = mat[(ox,oy)]
        o_diag_mask = o_diag[omask]
        # gap
        if(o_diag_mask.shape[0] == 0):
            decay[i] = 0
        else:
            decay[i] = o_diag_mask.mean()
        ox = np.delete(ox,-1)
        oy = np.delete(oy,0)
        omask = np.delete(omask,-1)

    ex = np.arange(chr_len)
    ey = np.arange(chr_len)
    except_mat = np.ones_like(mat,dtype = np.float32)
    for i in range(chr_len):
        if(decay[i] == 0):
            ex = np.delete(ex,-1)
            ey = np.delete(ey,0)
            continue
        except_mat[(ex,ey)] = decay[i]
        except_mat[(ey,ex)] = decay[i]
        ex = np.delete(ex,-1)
        ey = np.delete(ey,0)
        
    oe = mat/except_mat

    return oe

def GetCorr(oemat):
    cor_oe = np.corrcoef(oemat)
    cor_oe[np.isnan(cor_oe)] = 0

    return cor_oe

def minmax(mat):
    return (mat - np.min(mat)) / (np.max(mat) - np.min(mat))


filelist = ["H3K27ac","H3K4me1","H3K4me3","H3K9me3","H3K27me3","H3K36me3"]
res_tg = {'chr':[],"num":[]}

for chr in chrs_list:
    rdata = cooler.Cooler(micro_path)
    micro_mat = rdata.matrix(balance=True).fetch('chr'+chr)
    micro_mat[np.isnan(micro_mat)] = 0
    oe = GetOE(micro_mat)
    corr = GetCorr(oe)
    corr[np.isnan(corr)] = 0
    
    #######################################
    # nonezero = corr[np.nonzero(corr)]
    # cutv = np.percentile(nonezero,99)
    # corr[corr >= cutv] = cutv
    ########################################

    epic_l = []
    for f in filelist:
        seq = np.load(epi_path+f+".npz")['chr'+chr]
        seq = seq - np.nanmin(seq)
        seq[np.isnan(seq)] = 0
        nonzero = seq[np.nonzero(seq)]
        seq = np.log1p(seq)
        seq = minmax(seq)
        epic_l.append(seq)


    epic_mat = np.array(epic_l)
    epicxl,epicxr,microy = divideblock(corr,epic_mat,bsize=128)
    # print("chr "+chr+":",epicxl.shape,epicxr.shape,microy.shape)
    res_tg['chr'].append(chr)
    res_tg['num'].append(epicxl.shape[0])
    np.savez(store_dir+chr+".npz",xl = epicxl,xr = epicxr,y = microy)

pd.DataFrame(res_tg).to_csv("divide_log.csv")
