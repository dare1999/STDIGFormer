
import pandas as pd

import numpy as np



def get_train_testloc(data,historytimes,futuretimes):

    trainhisloc=[]
    trainfutureloc=[]
    for i in range(historytimes,data.shape[1]-futuretimes-1):
    
    
        trainhisloc.append(data[:,i-historytimes:i])
        
        trainfutureloc.append(data[:,i:i+futuretimes])
        
        


    testhisloc=data[:,data.shape[1]-futuretimes-historytimes:data.shape[1]-futuretimes]/min(data[:,data.shape[1]-futuretimes-historytimes:data.shape[1]-futuretimes].flatten())
    testfutureloc=data[:,data.shape[1]-futuretimes:]
    
    trainhisloc=np.array([_/4.3315 for _ in trainhisloc]).astype(np.float32)
    trainfutureloc=np.array([_ for _ in trainfutureloc]).astype(np.float32)

    
    return trainhisloc,trainfutureloc,testhisloc,testfutureloc
    
    
def get_train_testadjm(adjm,historytimes,futuretimes):

    trainadjm=[]
    for i in range(historytimes,adjm.shape[-1]-futuretimes-1):
    
        adjm_=adjm[:,:,i-historytimes:i]+np.eye(adjm.shape[0])[:,:,np.newaxis].astype(np.float32)
        adjm_=adjm_
        
        adjm_[adjm_==0]=-1#-1
        trainadjm.append(adjm_)
        
    testadjm=adjm[:,:,-futuretimes-historytimes:-futuretimes]+np.eye(adjm.shape[0])[:,:,np.newaxis].astype(np.float32)
    teatadjm=testadjm
    
    testadjm[testadjm==0]=-1#-1
    
    return trainadjm,testadjm
    

def get_train_testdata(historytimes=8,futuretimes=4):

    data=pd.read_csv('hotel.csv',header=None)#eth, hotel, univ, zara1, zara2
        
    data=np.array(data).astype(np.float32)

    trainhisx=[]
    
    trainhisy=[]
    
    trainfuturex=[]
    
    trainfuturey=[]

    
    locx=data[0::2,:]
   
    
    trainhisx,trainfuturex,testhisx,testfuturex=get_train_testloc(locx,historytimes,futuretimes)
    
    
    locy=data[1::2,:]
    
    trainhisy,trainfuturey,testhisy,testfuturey=get_train_testloc(locy,historytimes,futuretimes)
    
    
    
    traindatax=np.concatenate([trainhisx[...,np.newaxis],trainhisy[...,np.newaxis]],-1)
    
    traindatay=np.concatenate([trainfuturex[...,np.newaxis],trainfuturey[...,np.newaxis]],-1)
    
    testdatax=np.concatenate([testhisx[...,np.newaxis],testhisy[...,np.newaxis]],-1)
    
    testdatay=np.concatenate([testfuturex[...,np.newaxis],testfuturey[...,np.newaxis]],-1)
    
    adjm1=np.load('PtPf_hotel.npy').astype(np.float32)#graph3
    adjm1 = np.nan_to_num(adjm1)
    adjm1 = np.where(adjm1 == np.Inf, 1, adjm1)
    
    adjm2=np.load('PtPt_hotel.npy').astype(np.float32)#graph2
    adjm2 = np.nan_to_num(adjm2)
    adjm2 = np.where(adjm2 == np.Inf, 1, adjm2)

    adjm3=np.load('PtPw_hotel.npy').astype(np.float32)#graph1,#eth, hotel, univ, zara1, zara2
    adjm3 = np.nan_to_num(adjm3)
    adjm3=np.where(adjm3==np.Inf,1,adjm3)


    del data
    
    trainadjm1,testadjm1=get_train_testadjm(adjm1,historytimes,futuretimes)
    del adjm1
    trainadjm2,testadjm2=get_train_testadjm(adjm2,historytimes,futuretimes)
    del adjm2
    trainadjm3,testadjm3=get_train_testadjm(adjm3,historytimes,futuretimes)
    del adjm3
    trainadjm=[trainadjm1,trainadjm2,trainadjm3]
    del trainadjm1,trainadjm2,trainadjm3
    testadjm=[testadjm1,testadjm2,testadjm3]
    
    
    return traindatax,trainadjm,traindatay,testdatax,testadjm,testdatay
    
    
    
if __name__=='__main__':


    traindatax,trainadjm,traindatay,testdatax,testadjm,testdatay=get_train_testdata()