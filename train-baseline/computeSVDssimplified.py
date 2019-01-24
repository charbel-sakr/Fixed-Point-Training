import numpy as np

#baseFileNameJA = 'jacobians/actJacobain'
baseFileNameIn = 'jacobians/InJacobian'
baseFileNameOut = 'jacobians/OutJacobain'
SVDs = open('derivedStats/computedSVDS.txt','w')
for i in range(11):
    print(i)
    OutJac = np.load(baseFileNameOut+repr(i)+'at5.npy')
    _,SVs,_=np.linalg.svd(OutJac)
    maxSV = SVs.max()
    SVDs.write(repr(maxSV)+'\n')
    if((i>0)&(i<10)):
        InJac = np.load(baseFileNameIn+repr(i)+'at5.npy')
        _,SVs,_=np.linalg.svd(OutJac)
        maxSV = SVs.max()
        SVDs.write(repr(maxSV)+'\n')
SVDs.close()
