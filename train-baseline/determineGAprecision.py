import numpy as np
DRs = np.loadtxt('derivedStats/ADR.txt')
LSBs = np.loadtxt('derivedStats/ALSBs.txt')
precisionFile = open('derivedStats/Aprecisions.txt','w')
for i in range(20):
    DR = DRs[i]
    LSB = LSBs[i]
    precision = precision = np.log2(DR/LSB)+1
    precisionFile.write(repr(precision)+'\n')

