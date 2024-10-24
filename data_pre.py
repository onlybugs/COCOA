# Data pre.

import pyBigWig
import numpy as np
import os

##################################################################
chrs_list = ['12' ,'13' ,'14' ,'15' ,'16' ,'17' ,'18' ,'19']
in_resolution = 25000 # resolution
inp_dir = "---" # example as : ../data/epi/
out_dir = "---" # example as : ../data/hff1/
##################################################################

if(not os.path.exists(inp_dir)):
    print("Dir " + inp_dir + " is not exists! Check!")
    raise FileNotFoundError

if(not os.path.exists(out_dir)):
    os.makedirs(out_dir)

filelist = ["H3K27ac","H3K4me1","H3K4me3","H3K9me3","H3K27me3","H3K36me3"]
for fn in filelist:
    out_dict = {}
    for c in chrs_list:
        bw = pyBigWig.open(inp_dir + fn + ".bigWig")
        chrlen = bw.chroms("chr"+c)
        o = []
        for i in range(0,chrlen,in_resolution):
            end = i + in_resolution
            if(end >= chrlen): end = chrlen
            o.append(np.mean(np.array(bw.values("chr"+c,i,end))))
        out_dict['chr' + c] = np.array(o)
    np.savez(out_dir + fn + ".npz",**out_dict)

