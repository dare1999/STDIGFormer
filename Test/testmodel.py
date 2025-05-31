
import torch

from processdata import get_train_testdata

import torch.onnx
import numpy as np
import pandas as pd
import netron
import time

if __name__=='__main__':

    start_time = time.time()
    historytimes=8
    futuretimes=12
    traindatax,trainadjm,traindatay,testdatax,testadjm,testdatay=get_train_testdata(historytimes=historytimes,futuretimes=futuretimes)
    testdatax,  testdatay=traindatax[0],traindatay[0]
    adjms = [torch.tensor(_[0]).float() for _ in trainadjm]
    net=torch.load('net_epoch300.pth')
    net.eval()
    
    datax=torch.tensor(testdatax).float()
    adjms=[torch.tensor(_).float() for _ in testadjm]
    

    o=net(datax,adjms,returnlinearweights=False)
    
    o=o.reshape(o.shape[0],testdatay.shape[1],testdatay.shape[2])

    o=o.detach().numpy()
    
    o[testdatay==0]=0
    FDE=np.mean(np.sum((o[:,-1,:]-testdatay[:,-1,:])**2,-1)**0.5)
    ADE=np.mean(np.sum((o-testdatay)**2,-1)**0.5)
    
    print('FDE:',FDE)
    print('ADE:',ADE)
    
    result=np.zeros([o.shape[0]*2,testdatay.shape[1]])
    
    result[0::2,:]=o[:,:,0]
   
    result[1::2,:]=o[:,:,1]
    #
    result2 = np.zeros([testdatay.shape[0] * 2, o.shape[1]])
    result2[0::2] = testdatay[:, :, 0]
    result2[1::2] = testdatay[:, :, 1]
    result2 = pd.DataFrame(result2)
    result2.to_excel('resultzhenshi.xlsx', header=None, index=False)
    #
    result=pd.DataFrame(result)
    
    result.to_excel('result.xlsx',header=None,index=False)
    end_time = time.time()
    print("time: {:.2f} 秒".format(end_time - start_time))

