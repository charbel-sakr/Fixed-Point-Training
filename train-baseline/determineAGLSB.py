import numpy as np
WLSBs = np.loadtxt('derivedStats/WLSBs.txt')
WLSBs = np.delete(WLSBs,[9,16])
LSVs = np.loadtxt('derivedStats/computedSVDS.txt')
ALSBfile = open('derivedStats/ALSBs.txt','w')
for i in range(20):
    WLSB = WLSBs[i]
    LSV = LSVs[i] 
    ALSB = np.power(2.0,np.floor(np.log2(WLSB/np.sqrt(LSV))))
    ALSBfile.write(repr(ALSB)+'\n')
ALSBfile.close()
