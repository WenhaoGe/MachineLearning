import pandas as pd
import numpy as np
from numpy import zeros,ones
from pprint import pprint
import matplotlib.pyplot as plt
import random
import math
    
def calculateE(M, N):
    t = []
    x = np.repeat(random.uniform(0,1),N)
    x = x[...,None]
    phy = np.ones(N)[..., None]
    s1 = np.random.normal(0,0.3,N)
    for i in range (1,M+1):
        phy = np.concatenate((phy,x**i),axis=1)
        
    w = np.linalg.inv(phy.transpose() @ phy)
    for k in range(0,N):
        t.append(math.sin(2*math.pi*x[k])+s1[k])
    w = w @ phy.transpose() @ t
    j = t - phy @ w
    j = np.linalg.norm(j)
    e = math.sqrt(j/N)
    return e
    
arrTrain =[]
N = input("choose N for the training set(10 or 100)")
N = int(N)
for i in range(0,10):
    M = i
    e = calculateE(M,N)
    arrTrain.append(e)
    
arrTrain = np.array(arrTrain)[..., None]   

# generate array of e for test set
arrTest = []
N = 100
for i in range(0,10):
    M = i
    e = calculateE(M,N)
    arrTest.append(e)
    
arrTest = np.array(arrTest)[..., None]   
plt.plot(np.arange(0,10),arrTrain,'or-',label='train')
plt.plot(np.arange(0,10), arrTest,'ob-',label='test')
plt.legend()