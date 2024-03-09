import matplotlib.pyplot as plt  #画图用
import numpy as np


def pnas(input):
    alpha = 216.0
    n = 2.0
    k = 20.0
    beta = 1.0
    eta = 2.0
    ks0,ks1,kse = 1,0.01,0.03
    a,b,c,A,B,C,S = input
    # a,b,c,A,B,C,S,Se=input
    da,db,dc = -a+alpha/(1+C**n),-b+alpha/(1+A**n),-c+alpha/(1+B**n)+k*S/(1+S)
    dA,dB,dC = beta*(a-A),beta*(b-B),beta*(c-C)
    dS = -ks0*S+ks1*A-eta*(S-0)
    # dS,dSe=-ks0*S+ks1*A-eta*(S-Se),-kse*Se
    return np.array([da,db,dc,dA,dB,dC,dS])


n = 10000
dt = 0.02
dim = 7
np.random.seed(0)
x0 = np.random.uniform(0,1,dim)
X = np.zeros([n,dim])
X[0]=x0
for i in range(n-1):
    x = X[i]
    new_x = x+dt*pnas(x)
    X[i+1]=new_x

L = 715
# np.save('./data/train_pnas',X[-L:,:])
fig = plt.figure()
# X = np.load('./data/train_pnas.npy')
# ax = fig.gca(projection='3d')
# ax.plot(X[-1000:,3],X[-1000:,4],X[-1000:,5])
plt.plot(X[-L:,0],X[-L:,1])
# plt.plot(np.arange(len(X)),X[:,1])
# plt.ylim(0.6,0.725)
# plt.xlim(0.755,0.86)
plt.show()