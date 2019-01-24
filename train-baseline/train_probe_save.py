import theano
import theano.tensor as T
import numpy as np
import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import load
from theano.tensor.signal.pool import pool_2d
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

runningOutJacobian=[]
runningOutJacobian.append(layers.initRunningWeightJacobian(3,128))#res0

runningOutJacobian.append(layers.initRunningWeightJacobian(128,128))#res1
runningOutJacobian.append(layers.initRunningWeightJacobian(128,128))#res2
runningOutJacobian.append(layers.initRunningWeightJacobian(128,128))#res3

runningOutJacobian.append(layers.initRunningWeightJacobian(256,256))#res4
runningOutJacobian.append(layers.initRunningWeightJacobian(256,256))#res5
runningOutJacobian.append(layers.initRunningWeightJacobian(256,256))#res6

runningOutJacobian.append(layers.initRunningWeightJacobian(512,512))#res7
runningOutJacobian.append(layers.initRunningWeightJacobian(512,512))#res8
runningOutJacobian.append(layers.initRunningWeightJacobian(512,512))#res9

runningOutJacobian.append(layers.initRunningWeightJacobian(512,10))#res10

runningInJacobian=[]
runningInJacobian.append(layers.initRunningWeightJacobian(128,128))#res1
runningInJacobian.append(layers.initRunningWeightJacobian(128,128))#res2
runningInJacobian.append(layers.initRunningWeightJacobian(128,128))#res3
runningInJacobian.append(layers.initRunningWeightJacobian(128,256))#res4
runningInJacobian.append(layers.initRunningWeightJacobian(256,256))#res5
runningInJacobian.append(layers.initRunningWeightJacobian(256,256))#res6
runningInJacobian.append(layers.initRunningWeightJacobian(256,512))#res7
runningInJacobian.append(layers.initRunningWeightJacobian(512,512))#res8
runningInJacobian.append(layers.initRunningWeightJacobian(512,512))#res9

runningGradientStats=[]
res0=[]
res0.append(layers.initRunningGradStat())
runningGradientStats.append(res0)

res1=[]
res1.append(layers.initRunningGradStat())
res1.append(layers.initRunningGradStat())
runningGradientStats.append(res1)

res2=[]
res2.append(layers.initRunningGradStat())
res2.append(layers.initRunningGradStat())
runningGradientStats.append(res2)

res3=[]
res3.append(layers.initRunningGradStat())
res3.append(layers.initRunningGradStat())
runningGradientStats.append(res3)

res4=[]
res4.append(layers.initRunningGradStat())
res4.append(layers.initRunningGradStat())
res4.append(layers.initRunningGradStat())
runningGradientStats.append(res4)

res5=[]
res5.append(layers.initRunningGradStat())
res5.append(layers.initRunningGradStat())
runningGradientStats.append(res5)

res6=[]
res6.append(layers.initRunningGradStat())
res6.append(layers.initRunningGradStat())
runningGradientStats.append(res6)

res7=[]
res7.append(layers.initRunningGradStat())
res7.append(layers.initRunningGradStat())
res7.append(layers.initRunningGradStat())
runningGradientStats.append(res7)

res8=[]
res8.append(layers.initRunningGradStat())
res8.append(layers.initRunningGradStat())
runningGradientStats.append(res8)

res9=[]
res9.append(layers.initRunningGradStat())
res9.append(layers.initRunningGradStat())
runningGradientStats.append(res9)

res10=[]
res10.append(layers.initRunningGradStat())
runningGradientStats.append(res10)

runningActGrad=[]
res0=[]
res0.append(layers.initRunningGradStat())
runningActGrad.append(res0)

res1=[]
res1.append(layers.initRunningGradStat())
res1.append(layers.initRunningGradStat())
runningActGrad.append(res1)

res2=[]
res2.append(layers.initRunningGradStat())
res2.append(layers.initRunningGradStat())
runningActGrad.append(res2)

res3=[]
res3.append(layers.initRunningGradStat())
res3.append(layers.initRunningGradStat())
runningActGrad.append(res3)

res4=[]
res4.append(layers.initRunningGradStat())
res4.append(layers.initRunningGradStat())
runningActGrad.append(res4)

res5=[]
res5.append(layers.initRunningGradStat())
res5.append(layers.initRunningGradStat())
runningActGrad.append(res5)

res6=[]
res6.append(layers.initRunningGradStat())
res6.append(layers.initRunningGradStat())
runningActGrad.append(res6)

res7=[]
res7.append(layers.initRunningGradStat())
res7.append(layers.initRunningGradStat())
runningActGrad.append(res7)

res8=[]
res8.append(layers.initRunningGradStat())
res8.append(layers.initRunningGradStat())
runningActGrad.append(res8)

res9=[]
res9.append(layers.initRunningGradStat())
res9.append(layers.initRunningGradStat())
runningActGrad.append(res9)

res10=[]
res10.append(layers.initRunningGradStat())
runningActGrad.append(res10)

def resBlock(preAct,resParams,train,bn_updates):
    snrg=RandomStreams(12345)
    resActivations=[]
    current_params=resParams[0]
    inAct,newRM,newRV = layers.convBNAct(preAct,current_params,train)
    resActivations.append(inAct)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    inAct=layers.dropout(inAct, train, 0.8, snrg)

    current_params=resParams[1]
    outAct,newRM,newRV = layers.convBN(inAct,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    outAct = layers.slopedClipping(outAct+preAct)
    resActivations.append(outAct)
    return outAct,resActivations,bn_updates

def resBlockStride(preAct,resParams,train,bn_updates):
    snrg=RandomStreams(12345)
    resActivations=[]
    current_params=resParams[0]
    inAct,newRM,newRV = layers.convStrideBNAct(preAct,current_params,train)
    resActivations.append(inAct)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    inAct = layers.dropout(inAct,train,0.8,snrg)

    current_params=resParams[1]
    outAct,newRM,newRV = layers.convBN(inAct,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    current_params=resParams[2]
    shortCut,newRM,newRV = layers.convStrideBN(preAct,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))

    outAct = layers.slopedClipping(outAct+shortCut)
    resActivations.append(outAct)
    return outAct,resActivations,bn_updates

def feedForward(x, params, train):
    activations=[]
    bn_updates = []

    res0Params=params[0]
    res0Activations=[]
    current_params=res0Params[0]
    outAct,newRM,newRV = layers.convBNAct(x,current_params,train)
    bn_updates.append((current_params[4],newRM))
    bn_updates.append((current_params[5],newRV))
    res0Activations.append(outAct)
    activations.append(res0Activations)

    outAct,resActivations,bn_updates = resBlock(outAct,params[1],train,bn_updates)
    activations.append(resActivations)

    outAct,resActivations,bn_updates = resBlock(outAct,params[2],train,bn_updates)
    activations.append(resActivations)

    outAct,resActivations,bn_updates = resBlock(outAct,params[3],train,bn_updates)
    activations.append(resActivations)

    outAct,resActivations,bn_updates = resBlockStride(outAct,params[4],train,bn_updates)
    activations.append(resActivations)

    outAct,resActivations,bn_updates = resBlock(outAct,params[5],train,bn_updates)
    activations.append(resActivations)

    outAct,resActivations,bn_updates = resBlock(outAct,params[6],train,bn_updates)
    activations.append(resActivations)

    outAct,resActivations,bn_updates = resBlockStride(outAct,params[7],train,bn_updates)
    activations.append(resActivations)

    outAct,resActivations,bn_updates = resBlock(outAct,params[8],train,bn_updates)
    activations.append(resActivations)

    outAct,resActivations,bn_updates = resBlock(outAct,params[9],train,bn_updates)
    activations.append(resActivations)

    pooled = pool_2d(outAct,ws=(8,8),ignore_border=True,mode='average_exc_pad')
    pooled=pooled.flatten(2)

    res10Activations=[]
    res10Params=params[10]
    current_params=res10Params[0]
    z = layers.linOutermost(pooled,current_params)
    res10Activations.append(z)
    activations.append(res10Activations)
    #
    return z,bn_updates,activations

def mom(cost, params, learning_rate, runningGradientStats, activations, runningActGrad):
    updates = []

    resNumber=0
    for res in params:
        insideNumber=0
        for current_params in res:
            p_no = 0
            for p in current_params:#weight bias gamma beta
                p_no+=1
                if(p_no==4):
                    break
                g = T.grad(cost,p)
                updates.append((p, T.clip(p - learning_rate*g,-1.0,1.0)))
                #now update weight gradient stats
                if(p_no==1):
                    mu = T.mean(g)
                    sigma2 = T.var(g)
                    updates.append((runningGradientStats[resNumber][insideNumber][0],0.9*runningGradientStats[resNumber][insideNumber][0]+0.1*mu))
                    updates.append((runningGradientStats[resNumber][insideNumber][1],0.9*runningGradientStats[resNumber][insideNumber][1]+0.1*sigma2))
            insideNumber+=1
        resNumber+=1
    resNumber=0
    for res in activations:
        insideNumber=0
        for a in res:
            g=T.grad(cost,a)
            mu = T.mean(g)
            sigma2 = T.var(g)
            updates.append((runningActGrad[resNumber][insideNumber][0],0.9*runningActGrad[resNumber][insideNumber][0]+0.1*mu))
            updates.append((runningActGrad[resNumber][insideNumber][1],0.9*runningActGrad[resNumber][insideNumber][1]+0.1*sigma2))
            insideNumber+=1
        resNumber+=1
    return updates


def update_statistics(params,activations,runningInJacobian,runningOutJacobian):
    updates=[]
    for resNumber in range(11):
        #first out jacobians
        selector=1
        if(resNumber==0):
            selector=0
        if(resNumber==10):
            selector=0
        current_weight = params[resNumber]
        weight = current_weight[selector][0]
        activation = activations[resNumber][selector]
        axV=(0,2,3)
        if (resNumber==10):
            axV=0
        currentLayerVector = activation.mean(axis=axV)
        JW = T.grad(T.sum(currentLayerVector),weight)
        AVGW = T.sqr(JW)
        if(resNumber<10):
            AVGW = AVGW.mean(axis=(2,3))
        AVGJW = AVGW.flatten(2)
        if(resNumber==10):
            AVGJW =AVGJW.T
        updates.append((runningOutJacobian[resNumber],0.9*runningOutJacobian[resNumber]+0.1*AVGJW.T))
    for resNumber in range(1,10):
        weight = params[resNumber][0][0]
        activation = activations[resNumber][0]
        axV=(0,2,3)
        currentLayerVector = activation.mean(axis=axV)
        JW = T.grad(T.sum(currentLayerVector),weight)
        AVGW = T.sqr(JW)
        AVGW = AVGW.mean(axis=(2,3))
        AVGJW = AVGW.flatten(2)
        updates.append((runningInJacobian[resNumber-1],0.9*runningInJacobian[resNumber-1]+0.1*AVGJW.T))
    return updates


z, bn_updates, activations = feedForward(x, params, trainF)
y = T.argmax(z, axis=1)
cost = T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(z), t))
# compile theano functions
weight_updates = mom(cost, params, lr, runningGradientStats, activations, runningActGrad)
stat_updates = update_statistics(params,activations,runningInJacobian,runningOutJacobian)
predict = theano.function([x,trainF], y)
train = theano.function([x, t, lr, trainF], cost, updates=bn_updates+weight_updates)
stat_update =theano.function([x,trainF],[],updates=stat_updates)


batch_size = 250
# train model
LR=0.01
accFile = open('probedStats/accFile.txt','w')
gradStatFile = open('probedStats/gradStatFile.txt','w')
for i in range(200):
    if i==5:
        l=0
        for jac in runningOutJacobian:
            fname = 'jacobians/OutJacobain'+repr(l)+'at'+repr(i)+'.npy'
            np.save(fname,jac.get_value())
            l+=1
        l=0
        for jac in runningInJacobian:
            l+=1
            fname=  'jacobians/InJacobian'+repr(l)+'at'+repr(i)+'.npy'
            np.save(fname,jac.get_value())
    print('\n Starting Epoch ' + repr(i))
    if (i==100):
        LR *= 0.1
    #train*
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
        if start==0:
            stat_update(random_cropped,0.0)
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
    for res in runningGradientStats:
        for stat in res:
            gradStatFile.write(np.array_str(stat[0].get_value())+' ' +np.array_str(stat[1].get_value())+'\n')
    for res in runningActGrad:
        for stat in res:
            gradStatFile.write(np.array_str(stat[0].get_value())+' ' +np.array_str(stat[1].get_value())+'\n')

    gradStatFile.write('\n')
    print(np.array_str(total_loss)+' '+np.array_str(test_accuracy))
layers.saveParams('pretrained_params.save',params)
print("Params have been saved")
