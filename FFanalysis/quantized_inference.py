import theano
import theano.tensor as T
import numpy as np
import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import load
from theano.tensor.signal.pool import pool_2d
from six.moves import cPickle
x_train, t_train, x_test, t_test = load.cifar10(dtype=theano.config.floatX,grayscale=False)
labels = np.argmax(t_test,axis=1)
x_valid = x_train[:10000]
x_valid = x_valid.reshape((x_valid.shape[0],3,32,32))
x_test = x_test.reshape((x_test.shape[0],3,32,32))


# define symbolic Theano variables
x = T.tensor4()
#ireg = T.scalar()
B = T.scalar()
BA=T.fvector()
BW=T.fvector()
#prepare weight
#BC architecture is 2X256C3 - MP2 - 2x512C3 - MP2 - 2x512C3 - MP2 - 2x1024FC - 10
f = open('pretrained_params.save','rb')
params =[]
res0 = []
res0.append(cPickle.load(f))
params.append(res0)
res1=[]
res1.append(cPickle.load(f))
res1.append(cPickle.load(f))
params.append(res1)
res1=[]
res1.append(cPickle.load(f))
res1.append(cPickle.load(f))
params.append(res1)
res1=[]
res1.append(cPickle.load(f))
res1.append(cPickle.load(f))
params.append(res1)
res1=[]
res1.append(cPickle.load(f))
res1.append(cPickle.load(f))
res1.append(cPickle.load(f))
params.append(res1)
res1=[]
res1.append(cPickle.load(f))
res1.append(cPickle.load(f))
params.append(res1)
res1=[]
res1.append(cPickle.load(f))
res1.append(cPickle.load(f))
params.append(res1)
res1=[]
res1.append(cPickle.load(f))
res1.append(cPickle.load(f))
res1.append(cPickle.load(f))
params.append(res1)
res1=[]
res1.append(cPickle.load(f))
res1.append(cPickle.load(f))
params.append(res1)
res1=[]
res1.append(cPickle.load(f))
res1.append(cPickle.load(f))
params.append(res1)
res1=[]
res1.append(cPickle.load(f))
params.append(res1)

f.close()

def resBlock(preAct,resParams,train,BA1,BA2,BW1,BW2):
    current_params=resParams[0]
    current_params[0]=layers.quantizeWeight(current_params[0],BW1)
    inAct,_,_ = layers.convBNAct(preAct,current_params,train)
    inAct = layers.quantizeAct(inAct,BA1)
    inAct=0.8*inAct

    current_params=resParams[1]
    current_params[0]=layers.quantizeWeight(current_params[0],BW2)
    outAct,_,_ = layers.convBN(inAct,current_params,train)

    outAct = layers.slopedClipping(outAct+preAct)
    outAct = layers.quantizeAct(outAct,BA2)
    return outAct

def resBlockStride(preAct,resParams,train,BA1,BA2,BW1,BW2,BW3):
    current_params=resParams[0]
    current_params[0]=layers.quantizeWeight(current_params[0],BW1)
    inAct,_,_ = layers.convStrideBNAct(preAct,current_params,train)
    inAct=0.8*inAct

    current_params=resParams[1]
    current_params[0]=layers.quantizeWeight(current_params[0],BW2)
    outAct,_,_ = layers.convBN(inAct,current_params,train)

    current_params=resParams[2]
    current_params[0]=layers.quantizeWeight(current_params[0],BW3)
    shortCut,_,_ = layers.convStrideBN(preAct,current_params,train)

    outAct = layers.slopedClipping(outAct+shortCut)
    return outAct

def feedForward(x, params,B,BA,BW):
    train=0
    res0Params=params[0]
    current_params=res0Params[0]
    current_params[0]=layers.quantizeWeight(current_params[0],BW.take(0)+B)
    outAct,_,_ = layers.convBNAct(x,current_params,train)
    outAct=layers.quantizeAct(outAct,BA.take(1)+B)

    outAct = resBlock(outAct,params[1],train,BA.take(2)+B,BA.take(3)+B,BW.take(1)+B,BW.take(2)+B)

    outAct = resBlock(outAct,params[2],train,BA.take(4)+B,BA.take(5)+B,BW.take(3)+B,BW.take(4)+B)

    outAct = resBlock(outAct,params[3],train,BA.take(6)+B,BA.take(7)+B,BW.take(5)+B,BW.take(6)+B)

    outAct = resBlockStride(outAct,params[4],train,BA.take(8)+B,BA.take(9)+B,BW.take(7)+B,BW.take(8)+B,BW.take(9)+B)

    outAct = resBlock(outAct,params[5],train,BA.take(10)+B,BA.take(11)+B,BW.take(10)+B,BW.take(11)+B)

    outAct = resBlock(outAct,params[6],train,BA.take(12)+B,BA.take(13)+B,BW.take(12)+B,BW.take(13)+B)

    outAct = resBlockStride(outAct,params[7],train,BA.take(14)+B,BA.take(15)+B,BW.take(14)+B,BW.take(15)+B,BW.take(16)+B)

    outAct = resBlock(outAct,params[8],train,BA.take(16)+B,BA.take(17)+B,BW.take(17)+B,BW.take(18)+B)

    outAct = resBlock(outAct,params[9],train,BA.take(18)+B,BA.take(19)+B,BW.take(19)+B,BW.take(20)+B)

    pooled = pool_2d(outAct,ws=(8,8),ignore_border=True,mode='average_exc_pad')
    pooled=pooled.flatten(2)

    res10Params=params[10]
    current_params=res10Params[0]
    current_params[0]=layers.quantizeWeight(current_params[0],BW.take(21)+B)
    z = layers.linOutermost(pooled,current_params)
    #
    return z

z = feedForward(x, params,B,BA,BW)
y=T.argmax(z,axis=1)
# compile theano functions
predict = theano.function([x,B,BA,BW], y)
BAO = [  8., 5.,   4.,   4.,   3.,   3.,   3.,   3.,   3.,   4.,   4.,   4.,  4.,   3.,   3.,   4.,   3.,   2.,   1.,   0.]
BWO = [  11.,  10.,  10., 10.,  10.,  11.,  10.,  11.,  11.,  11.,  11.,  11.,  12.,  11., 11.,  11.,  11.,  11.,  10.,  10.,   9.,   7. ]

batch_size = 250
for b in range(20):
    running_acc=0.0
    batches=0.0
    for start in range(0,10000,batch_size):
        x_batch = x_test[start:start+batch_size]
        t_batch = labels[start:start+batch_size]
        running_acc += np.mean(t_batch==predict(x_batch,b,BAO,BWO))
        batches+=1
    test_acc = running_acc/batches
    print(repr(1.0-test_acc))
