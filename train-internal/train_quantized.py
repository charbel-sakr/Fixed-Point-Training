import theano
import theano.tensor as T
import numpy as np
import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import load
from theano.tensor.signal.pool import pool_2d
import quantizeGradPredicted as quantizeGrad
x_train, t_train, x_test, t_test = load.cifar10(dtype=theano.config.floatX,grayscale=False)
labels = np.argmax(t_test,axis=1)
x_train = x_train.reshape((x_train.shape[0],3,32,32))
x_test = x_test.reshape((x_test.shape[0],3,32,32))


# define symbolic Theano variables
x = T.tensor4()
t = T.matrix()
lr = T.scalar()
#ireg = T.scalar()
#selector = T.scalar()
trainF = T.scalar()

#prepare weight
#BC architecture is 2X256C3 - MP2 - 2x512C3 - MP2 - 2x512C3 - MP2 - 2x1024FC - 10
params =[]

res0 = []
res0.append(layers.initConvBN(3,128,32,3))
params.append(res0)

res1 = []
res1.append(layers.initConvBN(128,128,32,3))
res1.append(layers.initConvBN(128,128,32,3))
params.append(res1)

res2 = []
res2.append(layers.initConvBN(128,128,32,3))
res2.append(layers.initConvBN(128,128,32,3))
params.append(res2)

res3 = []
res3.append(layers.initConvBN(128,128,32,3))
res3.append(layers.initConvBN(128,128,32,3))
params.append(res3)

res4 = []
res4.append(layers.initConvBN(128,256,16,3))
res4.append(layers.initConvBN(256,256,16,3))
res4.append(layers.initConvBN(128,256,16,3))
params.append(res4)

res5=[]
res5.append(layers.initConvBN(256,256,16,3))
res5.append(layers.initConvBN(256,256,16,3))
params.append(res5)

res6=[]
res6.append(layers.initConvBN(256,256,16,3))
res6.append(layers.initConvBN(256,256,16,3))
params.append(res6)

res7=[]
res7.append(layers.initConvBN(256,512,8,3))
res7.append(layers.initConvBN(512,512,8,3))
res7.append(layers.initConvBN(256,512,8,3))
params.append(res7)

res8=[]
res8.append(layers.initConvBN(512,512,8,3))
res8.append(layers.initConvBN(512,512,8,3))
params.append(res8)

res9=[]
res9.append(layers.initConvBN(512,512,8,3))
res9.append(layers.initConvBN(512,512,8,3))
params.append(res9)

res10=[]
res10.append(layers.initLinOutermost(512,10))
params.append(res10)

def STEquant(x,b):
    return T.minimum(2.0-T.pow(2.0,1.0-b),layers.round3(x*T.pow(2.0,b-1.0))*T.pow(2.0,1.0-b))

def resBlock(preAct,resParams,train,bn_updates,BA,GAselect):
    snrg=RandomStreams(12345)
    current_params=resParams[0]
    inAct,newRM,newRV = layers.convBNAct(preAct,current_params[:6],train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    inAct=STEquant(inAct,BA)
    if GAselect==2:
        inAct=quantizeGrad.quantizeGradL2(inAct)
    elif GAselect==4:
        inAct=quantizeGrad.quantizeGradL4(inAct)
    elif GAselect==6:
        inAct=quantizeGrad.quantizeGradL6(inAct)
    elif GAselect==10:
        inAct=quantizeGrad.quantizeGradL10(inAct)
    elif GAselect==12:
        inAct=quantizeGrad.quantizeGradL12(inAct)
    elif GAselect==16:
        inAct=quantizeGrad.quantizeGradL16(inAct)
    elif GAselect==18:
        inAct=quantizeGrad.quantizeGradL18(inAct)

    inAct = layers.dropout(inAct,train,0.8,snrg)
    current_params=resParams[1]
    outAct,newRM,newRV = layers.convBN(inAct,current_params[:6],train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    outAct = layers.slopedClipping(outAct+preAct)
    return outAct,bn_updates

def resBlockStride(preAct,resParams,train,bn_updates,BA,GAselect):
    snrg=RandomStreams(12345)
    current_params=resParams[0]
    inAct,newRM,newRV = layers.convStrideBNAct(preAct,current_params[:6],train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    inAct=STEquant(inAct,BA)
    if GAselect==8:
        inAct=quantizeGrad.quantizeGradL8(inAct)
    elif GAselect==14:
        inAct=quantizeGrad.quantizeGradL14(inAct)

    inAct = layers.dropout(inAct,train,0.8,snrg)
    current_params=resParams[1]
    outAct,newRM,newRV = layers.convBN(inAct,current_params[:6],train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    current_params=resParams[2]
    shortCut,newRM,newRV = layers.convStrideBN(preAct,current_params[:6],train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    outAct = layers.slopedClipping(outAct+shortCut)
    return outAct,bn_updates


def feedForward(x, params, train):
    bn_updates = []
    BA=[8.,     8.,   7.,   7.,   6.,   6.,   6.,   6.,   6.,   7.,   7.,   7.,  7.,   6.,   6.,   7.,   6.,   5.,   4.,   3.]

    res0Params=params[0]
    current_params=res0Params[0]
    outAct,newRM,newRV = layers.convBNAct(x,current_params[:6],train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    outAct=STEquant(outAct,BA[1])
    outAct=quantizeGrad.quantizeGradL1(outAct)
    
    outAct,bn_updates = resBlock(outAct,params[1],train,bn_updates,BA[2],2)
    outAct=STEquant(outAct,BA[3])
    outAct=quantizeGrad.quantizeGradL3(outAct)

    outAct,bn_updates = resBlock(outAct,params[2],train,bn_updates,BA[4],4)
    outAct=STEquant(outAct,BA[5])
    outAct=quantizeGrad.quantizeGradL5(outAct)

    outAct,bn_updates = resBlock(outAct,params[3],train,bn_updates,BA[6],6)
    outAct=STEquant(outAct,BA[7])
    outAct=quantizeGrad.quantizeGradL7(outAct)

    outAct,bn_updates = resBlockStride(outAct,params[4],train,bn_updates,BA[8],8)
    outAct=STEquant(outAct,BA[9])
    outAct=quantizeGrad.quantizeGradL9(outAct)

    outAct,bn_updates = resBlock(outAct,params[5],train,bn_updates,BA[10],10)
    outAct=STEquant(outAct,BA[11])
    outAct=quantizeGrad.quantizeGradL11(outAct)

    outAct,bn_updates = resBlock(outAct,params[6],train,bn_updates,BA[12],12)
    outAct=STEquant(outAct,BA[13])
    outAct=quantizeGrad.quantizeGradL13(outAct)

    outAct,bn_updates = resBlockStride(outAct,params[7],train,bn_updates,BA[14],14)
    outAct=STEquant(outAct,BA[15])
    outAct=quantizeGrad.quantizeGradL15(outAct)

    outAct,bn_updates = resBlock(outAct,params[8],train,bn_updates,BA[16],16)
    outAct=STEquant(outAct,BA[17])
    outAct=quantizeGrad.quantizeGradL17(outAct)

    outAct,bn_updates = resBlock(outAct,params[9],train,bn_updates,BA[18],18)
    outAct=STEquant(outAct,BA[19])
    outAct=quantizeGrad.quantizeGradL19(outAct)

    pooled = pool_2d(outAct,ws=(8,8),ignore_border=True,mode='average_exc_pad')
    pooled=pooled.flatten(2)

    res10Params=params[10]
    current_params=res10Params[0]
    z = layers.linOutermost(pooled,current_params[:2])
    z=quantizeGrad.quantizeGradL20(z)
    #
    return z,bn_updates

def mom(cost, params, learning_rate):
    updates = []
    l=0
    B = [11.0,12.0,12.0,11.0,11.0,11.0,11.0,11.0,11.0,11.0,11.0,11.0,11.0,11.0,11.0,11.0,11.0,11.0,12.0,12.0,14.0,14.0]
    scale = [0.25,0.0625,0.0625,0.03125,0.03125,0.03125,0.03125,0.03125,0.03125,0.03125,0.03125,0.03125,0.03125,0.03125,0.015625,0.015625,0.015625,0.015625,0.015625,0.015625,0.015625,0.25]
    BW = [14.,  13.,  13., 13.,  13.,  14.,  13.,  14.,  14.,  14.,  14.,  14.,  15.,  14., 14.,  14.,  14.,  14.,  13.,  13.,   12.,   10.]
    BAcc = [9.,  13.,  13.,  13.,  13.,  12.,  13.,  12.,  12.,  12.,  12., 12.,  11.,  12.,  13.,  13.,  13.,  13.,  15.,  15.,  18.,  16.]
    DRAcc = [6.10351562e-05,   1.22070312e-04,   1.22070312e-04,  1.22070312e-04,   1.22070312e-04,   6.10351562e-05,  1.22070312e-04,   6.10351562e-05,   6.10351562e-05,    6.10351562e-05,   6.10351562e-05,   6.10351562e-05, 3.05175781e-05,   6.10351562e-05,   6.10351562e-05,  6.10351562e-05,   6.10351562e-05,   6.10351562e-05,  1.22070312e-04,   1.22070312e-04,   2.44140625e-04,   9.76562500e-04 ]
    for res in params:
        for current_params in res:
            p_no = 0
            for p in current_params:#weight bias gamma beta
                p_no+=1
                if(p_no==5):
                    break
                if((l==22)&(p_no==3)):
                    break
                g = T.grad(cost,p)
                #now update weight gradient stats
                if(p_no==1):
                    g = layers.quantizeNormalizedWeight(g,B[l],scale[l])
                    accumulator = current_params[-1]
                    nextAcc = accumulator -learning_rate*g
                    nextW = p+nextAcc
                    qW = layers.quantizeWeight(nextW,BW[l])
                    remainder = nextW - qW
                    updates.append((accumulator,layers.quantizeNormalizedWeight(remainder,BAcc[l],DRAcc[l])))
                    updates.append((p,qW))
                    l+=1
                else:
                    updates.append((p, T.clip(p - learning_rate*g,-1.0,1.0)))
    return updates



z, bn_updates  = feedForward(x, params, trainF)
y = T.argmax(z, axis=1)
cost = T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(z), t))
# compile theano functions
weight_updates = mom(cost, params, lr)
predict = theano.function([x,trainF], y)
train = theano.function([x, t, lr, trainF], cost, updates=bn_updates+weight_updates)


batch_size = 250
# train model
LR=0.01
accFile = open('accFileQuantizedPredictedPrecision.txt','w')


for i in range(200):
    print('\n Starting Epoch ' + repr(i))
    if (i==100):
        LR *= 0.1
    #train
    indices = np.random.permutation(50000)
    running_cost = 0.0
    batches = 0
    for start in range(0, 50000, batch_size):
        x_batch = x_train[indices[start:start + batch_size]]
        t_batch = t_train[indices[start:start + batch_size]]

        #horiz flip
        coins = np.random.rand(batch_size) < 0.5
        for r in range(batch_size):
            if coins[r]:
                x_batch[r,:,:,:] = x_batch[r,:,:,::-1]

        #random crop
        padded = np.pad(x_batch,((0,0),(0,0),(4,4),(4,4)),mode='constant')
        random_cropped = np.zeros(x_batch.shape, dtype=np.float32)
        crops = np.random.random_integers(0,high=8,size=(batch_size,2))
        for r in range(batch_size):
            random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]

        #train
        cost= train(random_cropped, t_batch, LR, 1)
        running_cost = running_cost + cost
        batches = batches+1
    total_loss = running_cost/batches

    # test
    running_accuracy =0.0
    batches = 0
    for start in range(0,10000,batch_size):
        x_batch = x_test[start:start+batch_size]
        t_batch = labels[start:start+batch_size]
        running_accuracy += np.mean(predict(x_batch,0) == t_batch)
        batches+=1
    test_accuracy = running_accuracy/batches
    accFile.write(np.array_str(total_loss)+' '+np.array_str(test_accuracy)+'\n')
    print(repr([np.array_str(total_loss),np.array_str(test_accuracy)]))
accFile.close()
