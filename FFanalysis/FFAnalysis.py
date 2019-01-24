import theano
import theano.tensor as T
import numpy as np
import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import load
from theano.tensor.signal.pool import pool_2d
from six.moves import cPickle
x_train, t_train, x_test, t_test = load.cifar100(dtype=theano.config.floatX)
labels = np.argmax(t_test,axis=1)
x_valid = x_train[:10000]
x_valid = x_valid.reshape((x_valid.shape[0],3,32,32))
x_test = x_test.reshape((x_test.shape[0],3,32,32))


# define symbolic Theano variables
x = T.tensor4()
#ireg = T.scalar()
#selector = T.scalar()

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

def resBlock(preAct,resParams,train):
    resActivations=[]
    current_params=resParams[0]
    inAct,_,_ = layers.convBNAct(preAct,current_params,train)
    resActivations.append(inAct)

    current_params=resParams[1]
    outAct,_,_ = layers.convBN(inAct,current_params,train)

    outAct = layers.slopedClipping(outAct+preAct)
    resActivations.append(outAct)
    return outAct,resActivations

def resBlockStride(preAct,resParams,train):
    resActivations=[]
    current_params=resParams[0]
    inAct,_,_ = layers.convStrideBNAct(preAct,current_params,train)
    resActivations.append(inAct)

    current_params=resParams[1]
    outAct,_,_ = layers.convBN(inAct,current_params,train)

    current_params=resParams[2]
    shortCut,_,_ = layers.convStrideBN(preAct,current_params,train)

    outAct = layers.slopedClipping(outAct+shortCut)
    resActivations.append(outAct)
    return outAct,resActivations

def feedForward(x, params):
    train=0
    activations=[]
    weights=[]

    res0Params=params[0]
    res0Activations=[]
    current_params=res0Params[0]
    outAct,_,_ = layers.convBNAct(x,current_params,train)
    res0Activations.append(outAct)
    activations.append(res0Activations[0])
    weights.append(current_params[0])

    outAct,resActivations = resBlock(outAct,params[1],train)
    weights.append(params[1][0][0])
    weights.append(params[1][1][0])
    activations.append(resActivations[0])
    activations.append(resActivations[1])

    outAct,resActivations = resBlock(outAct,params[2],train)
    weights.append(params[2][0][0])
    weights.append(params[2][1][0])
    activations.append(resActivations[0])
    activations.append(resActivations[1])

    outAct,resActivations = resBlock(outAct,params[3],train)
    weights.append(params[3][0][0])
    weights.append(params[3][1][0])
    activations.append(resActivations[0])
    activations.append(resActivations[1])

    outAct,resActivations = resBlockStride(outAct,params[4],train)
    weights.append(params[4][0][0])
    weights.append(params[4][1][0])
    weights.append(params[4][2][0])
    activations.append(resActivations[0])
    activations.append(resActivations[1])

    outAct,resActivations = resBlock(outAct,params[5],train)
    weights.append(params[5][0][0])
    weights.append(params[5][1][0])
    activations.append(resActivations[0])
    activations.append(resActivations[1])

    outAct,resActivations = resBlock(outAct,params[6],train)
    weights.append(params[6][0][0])
    weights.append(params[6][1][0])
    activations.append(resActivations[0])
    activations.append(resActivations[1])

    outAct,resActivations = resBlockStride(outAct,params[7],train)
    weights.append(params[7][0][0])
    weights.append(params[7][1][0])
    weights.append(params[7][2][0])
    activations.append(resActivations[0])
    activations.append(resActivations[1])

    outAct,resActivations = resBlock(outAct,params[8],train)
    weights.append(params[8][0][0])
    weights.append(params[8][1][0])
    activations.append(resActivations[0])
    activations.append(resActivations[1])

    outAct,resActivations = resBlock(outAct,params[9],train)
    weights.append(params[9][0][0])
    weights.append(params[9][1][0])
    activations.append(resActivations[0])
    activations.append(resActivations[1])

    pooled = pool_2d(outAct,ws=(8,8),ignore_border=True,mode='average_exc_pad')
    pooled=pooled.flatten(2)

    res10Params=params[10]
    current_params=res10Params[0]
    z = layers.linOutermost(pooled,current_params)
    weights.append(current_params[0])
    #
    z_fl = z.max(axis=1)
    y_fl = z.argmax(axis=1)
    evalues=[]
    print('got here')
    for activation in activations:
        E=0.0
        deriv_fl = T.grad(T.sum(z_fl),activation)
        for i in range(100):#should run over 100 but is too time consuming, use this approximation
            z_i = z.take(i,axis=1)
            deriv_i = T.grad(T.sum(z_i),activation)
            numerator = T.sqr(deriv_i - deriv_fl)
            denum = T.switch(T.eq(z_fl,z_i),1+0.0*z_i,T.sqr(z_i-z_fl))
            numerator = numerator.flatten()
            result = numerator/(denum.sum())
            E= E+T.sum(result)
        evalues.append(E/24)
    print('got here')
    for w in weights:
        E = 0.0
        deriv_fl_w = T.grad(z_fl.sum(),w)
        deriv_fl_w = deriv_fl_w.flatten()
        for i in range(10):
            z_i = z.take(i,axis=1)
            deriv_i_w = T.jacobian(z_i.sum(),w)
            deriv_i_w = deriv_i_w.flatten()
            numerator_w = T.sqr(deriv_i_w - deriv_fl_w)
            denum = T.switch(T.eq(z_fl,z_i),1+0.0*z_i,T.sqr(z_i-z_fl))
            result_w = numerator_w/(denum.sum())
            E = E+T.sum(result_w)
        evalues.append(E/24)
    print('got here')
    return evalues

evalues = feedForward(x, params)
# compile theano functions
compute = theano.function([x], evalues)
print('got here')
batch_size = 1
divider = 20000.0/batch_size
x_whole = np.concatenate((x_valid,x_test),axis=0)
Es=0.0
for i in range(20000):
    print('Reached '+repr(i))
    Es += np.asarray(compute(x_train[i:i+batch_size]))/batch_size
print(repr(Es/divider))

