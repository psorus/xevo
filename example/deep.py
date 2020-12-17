from xevo import eobj

import numpy as np
import json

from deeptools import *

datax,datay=gettestdata(f)


class deep(eobj):
  """neuronal network test"""
  def __init__(s,h=None,lr=0.001,batchsize=20,activation="relu",epochs=30):
    s.initial()
    
    if h is None:h=[]
    
    s.h=h#hidden layers
    s.lr=lr
    s.batchsize=batchsize
    s.activation=activation
    s.epochs=epochs
  def __str__(s):
    return json.dumps({"h":[int(q) for q in s.h],"lr":s.lr,"batchsize":s.batchsize,"activation":s.activation,"epochs":s.epochs,"strength":s.strength()},indent=2)
    
  def __add__(a,b):
    l=np.random.choice([np.min,np.max])([len(a.h),len(b.h)])
    l=[len(a.h),len(b.h)]
    l=np.random.randint(np.min(l),np.max(l)+1)
    ah=[int(q) for q in a.h]
    bh=[int(q) for q in b.h]
    while len(ah)<l:ah.append(bh[len(ah)])
    while len(bh)<l:bh.append(ah[len(bh)])
    h=[]
    for aa,bb in zip(ah,bh):
      h.append(np.random.choice([aa,bb]))
    return deep(h=h,lr=np.sqrt(a.lr*b.lr),batchsize=int(np.sqrt(a.batchsize*b.batchsize)),activation=np.random.choice([a.activation,b.activation]),epochs=(a.epochs+b.epochs)/2)



  def shallmaximize(s):return False

  
  def randomize(s):
    h=np.random.randint(1,26,(np.random.randint(0,7),))
    
    return deep(h=h,lr=np.exp(-3+-7*np.random.random()),batchsize=np.exp(np.log(400)*np.random.random()),activation=np.random.choice(["relu","sigmoid","tanh","linear"]),epochs=np.random.randint(2,100))
  

  def mutate(q):
    s=q.copy()#yes weird naming here, but I am lazy
    
    r=np.random.random()
    
    if r<0.5:
      #modify hidden
      if r<0.1:
        #add hidden
        if len(s.h)<6:s.h.append(np.random.randint(1,10))
      if r<0.2 and r>=0.1:
        #remove hidden
        if len(s.h)>0:del s.h[np.random.randint(len(s.h))]
      if r>=0.2:
        #chance hidden
        if len(s.h)>0:
          idd=np.random.randint(len(s.h))
          s.h[idd]+=np.random.normal(0,2,1)
          if s.h[idd]<=0:s.h[idd]=1
          if s.h[idd]>25:s.h[idd]=25
    else:
      #alter hyperparams
      if r<0.6:
        #chance lr
        if np.random.random()>0.5:
          s.lr*=1+np.random.random()
          if s.lr>1:s.lr=1.0
        else:
          s.lr/=1+np.random.random()
          if s.lr<0.000001:s.lr=0.000001
      if r<0.7 and r>=0.6:
        #chance batchsize
        if np.random.random()>0.5:
          s.batchsize*=1+np.random.random()
        else:
          s.batchsize/=1+np.random.random()
        s.batchsize=int(s.batchsize)
        if s.batchsize>400:s.batchsize=400
        if s.batchsize<1:s.batchsize=1

      if r<0.8 and r>=0.7:
        #chance activation
        s.activation=np.random.choice(["relu","sigmoid","tanh","linear"])
      if r<0.9 and r>=0.8:
        #chance epochs
        s.epochs+=np.random.randint(-5,6)
        if s.epochs<2:s.epochs=2
        if s.epochs>100:s.epochs=100
      if r>0.9:
        return s.randomize()
    
    return s
    
  def calcstrength(s):
    return traindensemodel(datax,datay,hidden=s.h,lr=s.lr,batchsize=s.batchsize,activation=s.activation,epochs=s.epochs)
  def _copy(s):
    return deep(h=[p for p in s.h],lr=s.lr,batchsize=s.batchsize,activation=s.activation,epochs=s.epochs)