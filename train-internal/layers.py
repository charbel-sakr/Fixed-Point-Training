import theano
import theano.tensor as T
import numpy as np

#from theano.tensor.nnet.abstract_conv import conv2d
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.bn import batch_normalization_train
from theano.tensor.nnet.bn import batch_normalization_test
from theano.tensor.signal.pool import pool_2d
from theano.ifelse import ifelse

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from six.moves import cPickle

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

class Round3(UnaryScalarOp):
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()
    def grad(self, inputs, gout):
        (gz,) = gout
        return gz,
round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)
def ste_binarize(x):
    return 2.*round3(T.clip((x+1.)/2.,0,1))-1

def convBNSTE(x, params, train):
    w,b,gamma,beta,RM,RV = params
    xc = conv2d(x,w) + b.dimshuffle('x',0,'x','x')
    xbn, newRM, newRV = batchNorm(xc, train, gamma, beta, RM, RV, 'spatial')
    xact = ste_binarize(xbn)
    return xact, newRM, newRV
def linBNSTE(x, params, train):
    w,b,gamma,beta,RM,RV = params
    xc = T.dot(x,w) + b
    xbn, newRM, newRV = batchNorm(xc, train, gamma, beta, RM, RV, (0,))
    xact = ste_binarize(xbn)
    return xact, newRM, newRV


def flipBits(binaryValues,p_f,snrg):
    corroptedBits = []
    for b in binaryValues:
        b_corropted = T.switch(snrg.binomial(size=b.shape,p=p_f),1.0-b,b)
        corroptedBits.append(b_corropted)
    return corroptedBits

def getActivationFromBinary(binaryValues,B):
    a=1.0*binaryValues[0]
    divider=0.5
    for i in range(1,B):
        a = a + divider*binaryValues[i]
        divider*=0.5
    return a

def getScaledWeightFromBinary(binaryValues,B,scale):
    return scale*getWeightValueFromBinary(binaryValues,B)
    #return scale*(getActivationFromBinary(binaryValues,B)-1)

def getWeightValueFromBinary(binaryValues,B):
    w = -1.0*binaryValues[0]
    divider = 0.5
    for i in range(1,B):
        w = w +divider*binaryValues[i]
        divider*=0.5
    return w

def getBinaryRepAct(a,B):
    binaryValues=[]
    a_temp=a
    for i in range(B):
        b_i = T.ge(a_temp,1)
        binaryValues.append(b_i)
        a_temp = a_temp - 1.0*b_i
        a_temp*=2
    return binaryValues

def getBinaryRepScaledWeight(w,B,scale):#let's say [-0.5 to 0.5] so scale = 0.5, first divide then multiply
    return getBinaryRepWeight(w/scale,B)
    #return getBinaryRepAct(w/scale+1,B)

def getBinaryRepWeight(w,B):
    binaryValues=[]
    MSB = T.lt(w,0)
    binaryValues.append(MSB)
    w_temp = w+MSB
    for i in range(1,B):
        w_temp*=2.0
        b_i = T.ge(w_temp,1.0)
        w_temp = w_temp - 1.0*b_i
        binaryValues.append(b_i)
    return binaryValues

def quantizeAct(x,B):
    return T.minimum(2.0-T.pow(2.0,1.0-B),T.round(x*T.pow(2.0,B-1.0),mode="half_away_from_zero")*T.pow(2.0,1.0-B))

def quantizeWeight(w,B):
    return T.minimum(1.0-T.pow(2.0,1.0-B),T.round(w*T.pow(2.0,B-1.0),mode="half_away_from_zero")*T.pow(2.0,1.0-B))

def quantizeNormalizedWeight(w,B,scale): #ref is -1 to 1
    return scale*T.minimum(1.0-T.pow(2.0,1.0-B),T.round((w/scale)*T.pow(2.0,B-1.0),mode="half_away_from_zero")*T.pow(2.0,1.0-B))

def quantizeNormalizedWeightStochastic(w,B,scale,srng): #ref is -1 to 1
    promoted = (w/scale)*T.pow(2.0,B-1.0)
    floored = T.floor(promoted)
    diff = promoted-floored
    toAdd = T.switch(srng.binomial(size=w.shape,p=diff),1.0,0.0)
    stochasticed = promoted+toAdd
    return scale*T.minimum(1.0-T.pow(2.0,1.0-B),stochasticed*T.pow(2.0,1.0-B))


def slopedClipping(x, m=1.0, alpha=2.0):
    return T.clip(x/m,0,alpha)

def batchNorm(x, train, gamma, beta, RM, RV, ax):
    values_train,_,_,newRM,newRV = batch_normalization_train(x,gamma,beta,axes=ax, running_mean=RM, running_var=RV)
    values = ifelse(T.neq(train,1),batch_normalization_test(x, gamma, beta, RM, RV, axes = ax),values_train)
    return values, newRM, newRV

def dropout(x, train, p_r, snrg):
    return ifelse(T.eq(train,1),T.switch(snrg.binomial(size=x.shape,p=p_r),x,0),p_r*x)

def convStrideBN(x, params, train, m=1.0, alpha=2.0):
    w,b,gamma,beta,RM,RV = params
    xc = conv2d(x,w,border_mode='half',subsample=(2,2)) + b.dimshuffle('x',0,'x','x')
    xbn, newRM, newRV = batchNorm(xc, train, gamma, beta, RM, RV, 'spatial')
    return xbn, newRM, newRV

def convStrideBNAct(x, params, train, m=1.0, alpha=2.0):
    w,b,gamma,beta,RM,RV = params
    xc = conv2d(x,w,border_mode='half',subsample=(2,2)) + b.dimshuffle('x',0,'x','x')
    xbn, newRM, newRV = batchNorm(xc, train, gamma, beta, RM, RV, 'spatial')
    xact = slopedClipping(xbn, m, alpha)
    return xact, newRM, newRV

def convBNAct(x, params, train, m=1.0, alpha=2.0):
    w,b,gamma,beta,RM,RV = params
    xc = conv2d(x,w,border_mode='half') + b.dimshuffle('x',0,'x','x')
    xbn, newRM, newRV = batchNorm(xc, train, gamma, beta, RM, RV, 'spatial')
    xact = slopedClipping(xbn, m, alpha)
    return xact, newRM, newRV

def convBN(x, params, train, m=1.0, alpha=2.0):
    w,b,gamma,beta,RM,RV = params
    xc = conv2d(x,w,border_mode='half') + b.dimshuffle('x',0,'x','x')
    xbn, newRM, newRV = batchNorm(xc, train, gamma, beta, RM, RV, 'spatial')
    return xbn, newRM, newRV

def linBNAct(x, params, train, m=1.0, alpha=2.0):
    w,b,gamma,beta,RM,RV = params
    xc = T.dot(x,w) + b
    xbn, newRM, newRV = batchNorm(xc, train, gamma, beta, RM, RV, (0,))
    xact = slopedClipping(xbn, m, alpha)
    return xact, newRM, newRV

def linOutermost(x,params):
    w,b = params
    return T.dot(x,w)+b

def initConvWeights(Nin, Nout, size, k):
    return theano.shared(np.asarray(np.random.uniform(low = -np.sqrt(6. / (Nin*size*size + Nout*size*size)), high = np.sqrt(6. / (Nin*size*size + Nout*size*size)),size=(Nout,Nin,k,k)), dtype = theano.config.floatX))

def initLinWeights(Nin, Nout):
    return theano.shared(np.asarray(np.random.uniform(low = -np.sqrt(6. / (Nout+Nin)), high = np.sqrt(6. / (Nout+Nin)),size=(Nin,Nout)), dtype = theano.config.floatX))

def initConvZero(Nin, Nout, size, k):
    return theano.shared(np.zeros(((Nout,Nin,k,k)), dtype = theano.config.floatX))

def initLinZero(Nin, Nout):
    return theano.shared(np.zeros(((Nin, Nout)), dtype = theano.config.floatX))

def initBias(Nout):
    return theano.shared(np.zeros((Nout,), dtype=theano.config.floatX))

def initBNGamma(N):
    return theano.shared(np.ones((N,), dtype=theano.config.floatX))

def initBNBeta(N):
    return theano.shared(np.zeros((N,), dtype=theano.config.floatX))

def initBNRM(N):
    return theano.shared(np.zeros((N,), dtype=theano.config.floatX))

def initBNRV(N):
    return theano.shared(np.zeros((N,), dtype=theano.config.floatX))

def initConvBN(Nin,Nout,size,k):
    params = []
    params.append(initConvWeights(Nin,Nout,size,k))
    params.append(initBias(Nout))
    params.append(initBNGamma(Nout))
    params.append(initBNBeta(Nout))
    params.append(initBNRM(Nout))
    params.append(initBNRV(Nout))
    params.append(initConvZero(Nin,Nout,size,k))
    return params

def initLinBN(Nin,Nout):
    params = []
    params.append(initLinWeights(Nin,Nout))
    params.append(initBias(Nout))
    params.append(initBNGamma(Nout))
    params.append(initBNBeta(Nout))
    params.append(initBNRM(Nout))
    params.append(initBNRV(Nout))
    params.append(initLinZero(Nin,Nout))
    return params

def initLinOutermost(Nin,Nout):
    params = []
    params.append(initLinWeights(Nin,Nout))
    params.append(initBias(Nout))
    params.append(initLinZero(Nin,Nout))
    return params

def saveParams(filename, params):
    f = open(filename, 'wb')
    for p_layer in params:
        for p in p_layer:
            cPickle.dump(p,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    return None

def initRunningActJacobian(Nin,Nout):
    return theano.shared(np.zeros((Nin,Nout), dtype=theano.config.floatX))

def initRunningWeightJacobian(Nin,Nout):
    return theano.shared(np.zeros((Nin,Nout), dtype=theano.config.floatX))

def initRunningGradStat():
    return [theano.shared(np.cast[theano.config.floatX](0.)),theano.shared(np.cast[theano.config.floatX](0.))]


def loadMNIST(filename):
    f = open(filename,'rb')
    params = []
    for i in range(4):
        current_params=[]
        current_params.append(cPickle.load(f))
        current_params.append(cPickle.load(f))
        params.append(current_params)

    return params

def loadParams(filename, NconvBN, NlinBN):
    f = open(filename, 'rb')
    params=[]
    #load convbn params: weight, bias, gamma, beta, rm, rv - 6 overall per layer
    for i in range(NconvBN):
        current_layer_params = []
        for j in range(6):
            current_layer_params.append(cPickle.load(f))
        params.append(current_layer_params)
    #load linbn params: same number
    for i in range(NlinBN):
        current_layer_params = []
        for j in range(6):
            current_layer_params.append(cPickle.load(f))
        params.append(current_layer_params)
    #load weight and bias for last layer
    last_layer_params = []
    last_layer_params.append(cPickle.load(f))
    last_layer_params.append(cPickle.load(f))
    params.append(last_layer_params)
    f.close()
    return params

def loadNormalizedParams(filename,Nlayers):
    f = open(filename, 'rb')
    params=[]
    for i in range(Nlayers):
        current_layer_params = []
        current_layer_params.append(cPickle.load(f))
        current_layer_params.append(cPickle.load(f))
        params.append(current_layer_params)
    f.close()
    return params
