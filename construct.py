
import numpy as np


train_list = ['1',"3" ,'5','7','9' ,'11' ,'13' ,'15' ,'17' ,'19']
val_list = ["18","20","21","22"]
inp_dir = "---" # From last step!
store_dir = "---"



trainxl = []
trainxr = []
trainy = []
for c in train_list:
    data = np.load(inp_dir + c + ".npz")
    xl = data['xl']
    xr = data['xr']
    y = data['y']
    hridx = np.random.choice(np.arange(y.shape[0]),int(y.shape[0]*1),replace=False)
    trainxl.append(xl[hridx])
    trainxr.append(xr[hridx])
    trainy.append(y[hridx])


xlo = np.concatenate(trainxl,axis=0)
xro = np.concatenate(trainxr,axis=0)
yo = np.concatenate(trainy,axis=0)
np.savez(store_dir + "train.npz",xl = xlo,xr = xro,y = yo)

testxl = []
testxr = []
testy = []
for c in val_list:
    data = np.load(inp_dir + c + ".npz")
    xl = data['xl']
    xr = data['xr']
    y = data['y']
    hridx = np.random.choice(np.arange(y.shape[0]),int(y.shape[0]*1),replace=False)
    testxl.append(xl[hridx])
    testxr.append(xr[hridx])
    testy.append(y[hridx])


xlo = np.concatenate(testxl,axis=0)
xro = np.concatenate(testxr,axis=0)
yo = np.concatenate(testy,axis=0)
np.savez(store_dir + "val.npz",xl = xlo,xr = xro,y = yo)

