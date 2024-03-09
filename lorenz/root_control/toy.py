import numpy as np
from scipy import integrate
import torch
import matplotlib.pyplot as plt
import math
import timeit
import torch.nn.functional as F
import torch.nn as nn
from functions import *
import torchsde

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

# class Ecoil(nn.Module):
#     def __init__(self):
#         super(Ecoil, self).__init__()
#         self.alpha = 216.0
#         self.n = 2.0
#         self.k = 20.0
#         self.beta = 1.0
#         self.eta = 2.0
#         self.ks0, self.ks1, self.kse = 1, 0.01, 0.03
#
#
#     def forward(self, t, state):
#         dstate = torch.zeros_like(state)
#         a,b,c,A,B,C,S = state[:,0],state[:,1],state[:,2],state[:,3],state[:,4],state[:,5],state[:,6],state[:,7]
#         dstate[:,0] = -a+alpha/(1+C**n)
#         dstate[:,1] = self.scale*(self.a2 * (y/self.scale) ** self.n / (self.s ** self.n + (y/self.scale) ** self.n)
#                                   + self.b2 * self.s ** self.n / (self.s ** self.n + (x/self.scale) ** self.n) - self.k * y/self.scale)
#         return dstate

class KimForger(nn.Module):

    def __init__(self,dim):
        super(KimForger, self).__init__()
        self.A = 0.05
        self.I = 0.003
        self.scale = 100.
        self.dim = dim
        self.noise_type = "scalar"
        self.sde_type = "ito"

    def forward(self, t, x):
        dx = torch.zeros_like(x)
        L = self.dim
        x,y,z = x[:,0:3*L:3],x[:,1:3*L:3],x[:,2:3*L:3]
        '''
        Kim Forger向量场
        '''
        dx[:,0:3*L:3] = self.scale*(torch.relu(1.-(z/self.scale)/self.A)-x/self.scale+self.I)
        dx[:,1:3*L:3] = x-y
        dx[:,2:3*L:3] = y-z
        # '''
        # 接来来计算耦合部分
        # '''
        # h_x = torch.zeros_like(x)
        # h_x[:,0:3*L:3] = self.sigma*x2
        # dx += torch.mm(torch.kron(self.A,torch.eye(3)),h_x.T).T
        return dx

    def g(self, t, x):
        dx = torch.zeros_like(x).unsqueeze(2)
        L = self.dim
        x, y, z = x[:, 0:3 * L:3], x[:, 1:3 * L:3], x[:, 2:3 * L:3]
        error_x = x[0].mean()-x.mean()
        error_y = y[0].mean() - y.mean()
        error_z = z[0].mean() - z.mean()
        dx[:, 0:3 * L:3,:] = self.scale * (self.I)*error_x
        return dx

dim = 10
true_y0 = torch.randn([1, dim * 3])  # 初值
# true_y0 = torch.randn(1,3).repeat(1,10)
# true_y0 = torch.tensor([[0.1,0.2,2.]])
# true_y0 = torch.tensor([[2.,2.,2.]])
t = torch.linspace(0., 20., 5000)  # 时间点
# ER = nx.random_graphs.erdos_renyi_graph(dim, p,seed=10) #生成ER图
# mask = torch.from_numpy(nx.to_numpy_array(ER))
# weight = Weight(mask)
# A = diag(weight.forward(delta))

func = KimForger(dim)
with torch.no_grad():
    true_y = odeint(func, true_y0, t, method='dopri5')[:,0,:]


batch_size, state_size, t_size = 1, 3*dim, 5000
sde = KimForger(dim)
ts = torch.linspace(0, 10, t_size)
# y0 = torch.full(size=(batch_size, state_size), fill_value=0.1)
sde_y0 = true_y[-2:-1,:]
with torch.no_grad():
    true_y = torchsde.sdeint(sde,sde_y0, ts, method='euler',names={'drift': 'forward'})[:,0,:]  # (t_size, batch_size, state_size) = (100, 3, 1).

fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(true_y[:,0],true_y[:,1],true_y[:,2])
# ax.plot(true_y[:,3],true_y[:,4],true_y[:,5])
for i in range(dim):
    plt.plot(np.arange(len(true_y)),true_y[:,i*3])
# plt.plot(np.arange(len(true_y)),true_y[:,3])
# plt.plot(t,true_y[:,2])
plt.show()

