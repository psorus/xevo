import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def f(x):
  return x[0]+x[1]*x[2]+np.sin(x[3])

def gettestdata(f,n=1000):
  x=np.random.normal(0,1,(n,4))
  y=np.array([[f(xx)] for xx in x])
  return x,y

def traindensemodel(datax,datay,hidden,lr=0.001,batchsize=20,activation="relu",epochs=30):
  try:
    batchsize=int(batchsize)
    print("training model with",hidden,lr,batchsize,activation,epochs)
    # exit()
    # print("training model")
    inputs=keras.Input(shape=datax.shape[1:])
    x=inputs
    for h in hidden:
      x=layers.Dense(h,activation=activation)(x)
    x=layers.Dense(datay.shape[-1])(x)
    model=keras.Model(inputs=inputs,outputs=x,name="super_simple_model")
    model.compile(loss="mse",optimizer=keras.optimizers.Adam(lr))
    clen=int(datax.shape[0])
    vlen=int(clen*0.2)
    history=model.fit(datax[:-vlen],datay[:-vlen],batch_size=batchsize,epochs=epochs,validation_split=0.2,verbose=0)
    score=model.evaluate(datax[-vlen:],datay[-vlen:],verbose=0)
    return score
  except:
    print("failed")
    return 1000.0
    # return traindensemodel(datax,datay,hidden,lr,batchsize,activation,epochs)
    
  
  
if __name__=="__main__":
  datax,datay=gettestdata(f)
  print(traindensemodel(datax,datay,[3,2]))
  import time
  t0=time.time()
  
  for i in range(10):
    print(traindensemodel(datax,datay,[3,2]))

  
  t1=time.time()
  print("!",(t1-t0)/10)

