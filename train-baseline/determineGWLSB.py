import numpy as np
gradStats = np.loadtxt('probedStats/gradStatFile.txt')
DRs = np.loadtxt('derivedStats/WDR.txt')
gradStats=gradStats.reshape((200,42, 2))

#print(index)
LSBfile = open("derivedStats/WLSBs.txt",'w')
for i in range(22):
    DR = DRs[i]
    minVal = np.sqrt(np.min(gradStats[:,i,1]))#correction factor very close to 1
    LSBW = np.power(2.0,np.floor(np.log2(0.25*minVal)))
    LSBfile.write(repr(LSBW)+'\n')

