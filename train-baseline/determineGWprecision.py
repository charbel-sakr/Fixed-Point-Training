import numpy as np
DRs = np.loadtxt('derivedStats/WDR.txt')
LSBs = np.loadtxt('derivedStats/WLSBs.txt')
precisionFile = open('derivedStats/Wprecisions.txt','w')
for i in range(22):
    DR = DRs[i]
    LSB = LSBs[i]
    precision = precision = np.log2(DR/LSB)+1
    precisionFile.write(repr(precision)+'\n')

