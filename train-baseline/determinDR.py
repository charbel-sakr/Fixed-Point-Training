import numpy as np

gradStats = np.loadtxt('probedStats/gradStatFile.txt')
gradStats=gradStats.reshape((200,42, 2))
WDR = open("derivedStats/WDR.txt",'w')
ADR = open("derivedStats/ADR.txt",'w')
for i in range(22):
    sigmaMax = np.max(np.sqrt(gradStats[:,i,1]))
    t = np.argmax(gradStats[:,i,1])
    correct = sigmaMax/np.sqrt(1.0-0.9)
    DR = np.power(2.0,np.ceil(np.log2(2.0*correct)))
    WDR.write(repr(DR)+'\n')
for i in range(22,42):
    sigmaMax = np.max(np.sqrt(gradStats[:,i,1]))
    t = np.argmax(gradStats[:,i,1])
    correct = sigmaMax/np.sqrt(1.0-0.9)#activations are more susceptible
    DR = np.power(2.0,np.ceil(np.log2(8.0*correct)))
    ADR.write(repr(DR)+'\n')

