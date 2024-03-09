import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import mod
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_normalization import SpectralNorm
import math
import timeit
from torchdiffeq import odeint, odeint_adjoint
import torchsde

torch.set_default_dtype(torch.float64)

def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)

class ICNN(nn.Module):
    def __init__(self, input_shape, layer_sizes, smooth_relu_thresh=0.1,eps=1e-3):
        super(ICNN, self).__init__()
        self._input_shape = input_shape
        self._layer_sizes = layer_sizes
        self._eps = eps
        self._d = smooth_relu_thresh
        # self._activation_fn = activation_fn
        ws = []
        bs = []
        us = []
        prev_layer = input_shape
        w = torch.empty(layer_sizes[0], *input_shape)
        nn.init.xavier_normal_(w)
        ws.append(nn.Parameter(w))
        b = torch.empty([layer_sizes[0], 1])
        nn.init.xavier_normal_(b)
        bs.append(nn.Parameter(b))
        for i in range(len(layer_sizes))[1:]:
            w = torch.empty(layer_sizes[i], *input_shape)
            nn.init.xavier_normal_(w)
            ws.append(nn.Parameter(w))
            b = torch.empty([layer_sizes[i], 1])
            nn.init.xavier_normal_(b)
            bs.append(nn.Parameter(b))
            u = torch.empty([layer_sizes[i], layer_sizes[i - 1]])
            nn.init.xavier_normal_(u)
            us.append(nn.Parameter(u))
        self._ws = nn.ParameterList(ws)
        self._bs = nn.ParameterList(bs)
        self._us = nn.ParameterList(us)

    def smooth_relu(self, x):
        relu = x.relu()
        # TODO: Is there a clean way to avoid computing both of these on all elements?
        # sq = (2 * self._d * relu.pow(3) - relu.pow(4)) / (2 * self._d ** 3)
        # sq = relu.pow(2)/(2*self._d)
        sq = (2 * self._d * relu.pow(3) - relu.pow(4)) / (2 * self._d ** 3)
        lin = x - self._d / 2
        return torch.where(relu < self._d, sq, lin)

    def dsmooth_relu(self, x):
        relu = x.relu()
        # TODO: Is there a clean way to avoid computing both of these on all elements?
        sq = (6 * self._d * relu.pow(2) - 4*relu.pow(3)) / (2 * self._d ** 3)
        # sq = relu/self._d
        lin = 1.
        return torch.where(relu < self._d, sq, lin)

    def icnn_fn(self, x):
        # x: [batch, data]
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        else:
            data_dims = list(range(1, len(self._input_shape) + 1))
            x = x.permute(*data_dims, 0)
        z = self.smooth_relu(torch.addmm(self._bs[0], self._ws[0], x))
        # print('--------------------',self._ws[0].shape)
        for i in range(len(self._us)):
            u = F.softplus(self._us[i])
            w = self._ws[i + 1]
            b = self._bs[i + 1]
            z = self.smooth_relu(torch.addmm(b, w, x) + torch.mm(u, z))
        return z

    def inter_dicnn_fn(self, x):
        # x: [batch, data]
        x = x.clone().detach()
        N,dim = x.shape[0],x.shape[1]
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        else:
            data_dims = list(range(1, len(self._input_shape) + 1))
            x = x.permute(*data_dims, 0)
        z = torch.addmm(self._bs[0], self._ws[0], x)
        dz = self.dsmooth_relu(z).unsqueeze(2).repeat(1,1,dim)*self._ws[0].unsqueeze(1).repeat(1,N,1)
        for i in range(len(self._us)):
            u = F.softplus(self._us[i])
            w = self._ws[i + 1]
            b = self._bs[i + 1]
            # print(u.shape, w.shape, b.shape, z.shape)
            z = torch.addmm(b, w, x) + torch.mm(u, self.smooth_relu(z))
            for k in range(dim):
                dz[:,:,k] = torch.mm(u,dz[:,:,k])
            dz = self.dsmooth_relu(z).unsqueeze(2).repeat(1,1,dim)*w.unsqueeze(1).repeat(1,N,1) \
                + self.dsmooth_relu(z).unsqueeze(2).repeat(1,1,dim)*dz
        return dz

    def dicnn_fn(self,x):
        dim = x.shape[1]
        target = self.target.repeat(len(x), 1)
        z = self.icnn_fn(x)
        z0 = self.icnn_fn(target)
        dregular = 2*self._eps*(x-target)
        dz = self.dsmooth_relu(z-z0).unsqueeze(2).repeat(1,1,dim)*self.inter_dicnn_fn(x)+dregular
        return dz[0]

    def forward(self,x):
        target = torch.zeros_like(x)
        z = self.icnn_fn(x)
        z0 = self.icnn_fn(target)
        regular = self._eps * (x-target).pow(2).sum(dim=1).view(-1,1)
        return self.smooth_relu(z-z0).T+regular

def lya(ws,bs,us,smooth,x,input_shape):
    if len(x.shape) < 2:
        x = x.unsqueeze(0)
    else:
        data_dims = list(range(1, len(input_shape) + 1))
        x = x.permute(*data_dims, 0)
    z = smooth(torch.addmm(bs[0],ws[0], x))
    for i in range(len(us)):
        u = F.softplus(us[i])
        w = ws[i + 1]
        b = bs[i + 1]
        z = smooth(torch.addmm(b, w, x) + torch.mm(u, z))
    return z

def dlya(ws,bs,us,smooth_relu,dsmooth_relu,x,input_shape):
    N, dim = x.shape[0], x.shape[1]
    if len(x.shape) < 2:
        x = x.unsqueeze(0)
    else:
        data_dims = list(range(1, len(input_shape) + 1))
        x = x.permute(*data_dims, 0)
    z = torch.addmm(bs[0], ws[0], x)
    dz = dsmooth_relu(z).unsqueeze(2).repeat(1, 1, dim) * ws[0].unsqueeze(1).repeat(1, N, 1)
    for i in range(len(us)):
        u = F.softplus(us[i])
        w = ws[i + 1]
        b = bs[i + 1]
        # print(u.shape, w.shape, b.shape, z.shape)
        z = torch.addmm(b, w, x) + torch.mm(u, smooth_relu(z))
        for k in range(dim):
            dz[:, :, k] = torch.mm(u, dz[:, :, k])
        dz = dsmooth_relu(z).unsqueeze(2).repeat(1, 1, dim) * w.unsqueeze(1).repeat(1, N, 1) \
             + dsmooth_relu(z).unsqueeze(2).repeat(1, 1, dim) * dz
    return dz

class ControlNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output,dim):
        super(ControlNet, self).__init__()
        self.layer1 = SpectralNorm(torch.nn.Linear(n_input, n_hidden))
        self.layer2 = SpectralNorm(torch.nn.Linear(n_hidden, n_hidden))
        self.layer3 = SpectralNorm(torch.nn.Linear(n_hidden, n_output))
        self.dim = dim
        self.upperbound = 1.8
        self.lowerbound = -self.upperbound

    def function(self, x):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out

    def bound(self, x):
        # relu = (x-self.lowerbound).relu()+self.lowerbound
        # relu = self.upperbound-(self.upperbound-relu).relu()
        # return torch.where(relu < self.upperbound, relu, self.upperbound)
        x1 = torch.where(x < self.upperbound, x, 0.)
        return torch.where(self.lowerbound < x, x1, 0.)

    def forward(self, x):
        target = torch.zeros_like(x)
        u = self.function(x)
        return self.bound(u*x)


class Model_lorenz(nn.Module):

    def __init__(self,mode,n_input, n_hidden, n_output,dim,R,input_shape,layer_sizes=[64, 64,1],smooth_relu_thresh=0.1, eps=1e-3):
        super(Model_lorenz, self).__init__()

        self.sigma = 10.
        self.rho = 28.
        self.beta = 8/3

        self.scale = 1.
        self.dim = dim
        self._eps = eps
        self.input_shape = input_shape
        self._control = ControlNet(n_input, n_hidden, 3,self.dim)
        self._lya = ICNN(input_shape, layer_sizes, smooth_relu_thresh,eps)
        self.noise_type = "scalar"
        self.sde_type = "ito"
        self.R = R # 同步流形
        self.mode = mode

    def lorenz(self,t,state):
        dx = torch.zeros_like(state)
        L = self.dim
        x,y,z = state[:,0:L],state[:,L:2*L],state[:,2*L:3*L]
        '''
        Lorenz: drive-response
        '''
        # 向量场赋值
        dx[:, 0*L:1*L] = self.sigma*(y-x)
        dx[:, 1*L:2*L] = self.rho*x-y-x*z*self.scale
        dx[:, 2*L:3*L] = x*y*self.scale-self.beta*z
        return dx



    def forward(self, t, state):
        dstate = torch.zeros_like(state)
        L = self.dim
        x, y, z = state[:, 0:L], state[:, L:2 * L], state[:, 2 * L:3 * L]
        dstate[:, 0*L:1*L] = self.sigma*(y-x)
        dstate[:, 1*L:2*L] = self.rho*x-y-x*z*self.scale
        dstate[:, 2*L:3*L] = x*y*self.scale-self.beta*z
        return dstate


    def forward_nonauto(self, t, state):
        dstate = torch.zeros_like(state)
        L = self.dim
        m = len(state)
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        x_s, y_s, z_s= self.R[0:m,0],self.R[0:m,1],self.R[0:m,2]
        dstate[:, 0] = self.sigma*(y-x)
        dstate[:, 1] = self.rho*x-y-self.scale*(x+x_s)*(z+z_s)+self.scale*x_s*z_s
        dstate[:, 2] = self.scale*(x+x_s)*(y+y_s)-self.scale*x_s*y_s-self.beta*z
        return dstate



    def train_g(self, t, state):
        dstate = torch.zeros_like(state)
        L = self.dim
        m = len(state)
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        x_s, y_s, z_s = self.R[0:m, 0], self.R[0:m, 1], self.R[0:m, 2]

        u = self._control(state)
        if self.mode == 1:
            dstate[:, 0] = u[:, 0] * (y-x+y_s-x_s) # control sigma
        if self.mode == 2:
            dstate[:, 1] = u[:, 1] * (x + x_s) # control rho
        if self.mode == 3:
            dstate[:, 2] = -u[:, 2] * (z+z_s) # control beta
        if self.mode == 4:
            dstate[:, 0] = u[:, 0] * (y - x + y_s - x_s)  # control sigma
            dstate[:, 1] = u[:, 1] * (x + x_s)  # control rho
        if self.mode == 5:
            dstate[:, 0] = u[:, 0] * (y - x + y_s - x_s)  # control sigma
            dstate[:, 2] = -u[:, 2] * (z + z_s)  # control beta
        if self.mode == 6:
            dstate[:, 1] = u[:, 1] * (x + x_s)  # control rho
            dstate[:, 2] = -u[:, 2] * (z + z_s)  # control beta
        if self.mode == 7:
            dstate[:, 0] = u[:, 0] * (y - x + y_s - x_s)  # control sigma
            dstate[:, 1] = u[:, 1] * (x + x_s)  # control rho
            dstate[:, 2] = -u[:, 2] * (z + z_s)  # control beta

        # u = self._control(torch.cat((a,b,c),dim=1))
        # dstate[:, 0] = u[:,0]#*(y-x+y_s-x_s)
        # dstate[:, 1] = u[:, 1] #* (x + x_s)
        # dstate[:, 2] = u[:, 2]
        # dstate[:, 1] = self._control.bound(u[:,1]*(x+x_s))
        # dstate[:, 2] = self._control.bound(u[:,2]*(z+z_s))
        # return u.unsqueeze(2)
        # return -50*state.unsqueeze(2)
        return dstate.unsqueeze(2)

    def lorenz_g(self, t, state):
        dstate = torch.zeros_like(state)
        st = torch.zeros([len(state),3])
        L = self.dim
        m = len(state)
        x,y,z = state[:,0:L],state[:,L:2*L],state[:,2*L:3*L]
        st[:, 0] = x[:,1]-x[:,0]
        st[:, 1] = y[:,1]-y[:,0]
        st[:, 2] = z[:,1]-z[:,0]
        # if torch.linalg.norm(st,2)<10.:
        u = self._control(st)
        # else:
        #     u = self._control(torch.zeros_like(st))
        if self.mode == 1:
            dstate[:, 1] = u[:, 0] * (y[:,1] - x[:,1])  # control sigma
        if self.mode == 2:
            dstate[:, 3] = u[:, 1] * x[:,1]   # control rho
        if self.mode == 3:
            dstate[:, 5] = -u[:, 2] * z[:,1]  # control beta
        if self.mode == 4:
            dstate[:, 1] = u[:, 0] * (y[:,1] - x[:,1])  # control sigma
            dstate[:, 3] = u[:, 1] * x[:,1]  # control rho
        if self.mode == 5:
            dstate[:, 1] = u[:, 0] * (y[:,1] - x[:,1])  # control sigma
            dstate[:, 5] = -u[:, 2] * z[:,1]   # control beta
        if self.mode == 6:
            dstate[:, 3] = u[:, 1] * x[:,1]   # control rho
            dstate[:, 5] = -u[:, 2] * z[:,1]   # control beta
        if self.mode == 7:
            dstate[:, 1] = u[:, 0] * (y[:,1] - x[:,1])  # control sigma
            dstate[:, 3] = u[:, 1] * x[:,1] # control rho
            dstate[:, 5] = -u[:, 2] * z[:,1]   # control beta

        # u = self._control(torch.cat((a,b,c),dim=1))
        # dstate[:, 3] = u[:, 0]#*(y[:,1]-x[:,1])
        # dstate[:, 4] = u[:, 1] #* x[:, 1]
        # dstate[:, 5] = u[:, 2]
        # dstate[:, 4] = self._control.bound(u[:, 1]*x[:,1])
        # dstate[:, 5] = self._control.bound(u[:, 2]*z[:,1])
        # dstate[:,4] = self._control.bound(5*(st[:,1]+st[:,0]+st[:,2]))# 根节点为y变量，对根节点进行控制
        # dstate[:, 3:6] = self._control.bound(10*st)
        # return u.unsqueeze(2)
        # return -50*state.unsqueeze(2)
        return dstate.unsqueeze(2)



def generate(model,true_y0,mode,g_case='lorenz_g'):
    # model._control.load_state_dict(torch.load('./data/control_y_50_2000_2.1_leaf.pkl'))
    N = 10000
    m = 50
    t = torch.linspace(0., 10., N)  # 时间点
    data = torch.zeros([m,  N, 6])
    for i in range(m):
        setup_seed(i)
        noise = torch.Tensor(1, 3).uniform_(-1.5, 1.5)
        true_y0[:, 0:6:2] += noise
        with torch.no_grad():
            cont_y = torchsde.sdeint(model, true_y0, t, method='euler',
                                     names={'drift': 'lorenz', 'diffusion':g_case})[:,
                     0, :]
            data[i, :N, :] = cont_y
            # cont_y = torchsde.sdeint(model, cont_y[-2:-1, :], t, method='euler',
            #                          names={'drift': 'lorenz', 'diffusion': g_case})[:,
            #          0, :]
            # data[i, N:2*N, :] = cont_y
            # cont_y = torchsde.sdeint(model, cont_y[-2:-1, :], t, method='euler',
            #                          names={'drift': 'lorenz', 'diffusion': g_case})[:,
            #          0, :]
            # data[i,2*N:3*N, :] = cont_y
        print('current iteration:{}'.format(i))
    torch.save(data, './data/data_para_{}_AI_{}_{}.pt'.format(mode,model._control.upperbound,m))


def save_loss():
    loss_list = []
    for i in range(7):
        data = torch.load('./data/loss_AIpara_{}.pt'.format(i+1))
        loss_list.append(data[-1].item())
    np.save('./data/AIpara_loss',np.array(loss_list))
# save_loss()

def success_rate(data,T=1000,threshold=1.0):
    state1,state2 = data[:,:,0:6:2],data[:,:,1:6:2]
    N = data.shape[0]
    error = torch.sqrt(torch.sum((state1-state2)**2,dim=2))
    mse = torch.mean(error[:,-T:],dim=1)
    succ = torch.where(mse<threshold)
    # print(succ[0])
    return succ[0] #len(succ[0])/N

def generate_success_rate():
    def get_succ_seed():
        succ_list = []
        for j in range(7):
            for str in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
                data = torch.load('./data/data_para_{}_AI_{}_50.pt'.format(j + 1, str))
                succ = success_rate(data)
                for s in succ:
                    if s in succ_list:
                        pass
                    else:
                        succ_list.append(s.item())
        return sorted(succ_list)
    succ_list = get_succ_seed()
    succ_rate = []
    for j in range(7):
        sub_succ_rate = []
        for str in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
            data = torch.load('./data/data_para_{}_AI_{}_50.pt'.format(j + 1, str))
            data = data[succ_list]
            succ = success_rate(data)
            sub_succ_rate.append(len(succ)/len(succ_list))
        succ_rate.append(sub_succ_rate)
    return succ_rate

# succ_rate = generate_success_rate()
# np.save('succ_rate',succ_rate)
#[244/255,241/255,185/255]
marker_list = ['s','v','o','D','*','p','^']
seven_color = [[75/255,102/255,173/255],[98/255,190/255,166/255],
               [205/255,234/255,157/255],'gold',
               [253/255,186/255,107/255],[235/255,96/255,70/255],[163/255,6/255,67/255]
               ]
# seven_color = [[105/255,38/255,5/255],[169/255,59/255,3/255],
#                [225/255,100/255,14/255],[252/255,149/255,39/255],
#                [254/255,205/255,97/255],[254/255,237/255,167/255],[255/255,253/255,223/255]
#                ][::-1]

# succ_rate = np.array(succ_rate)+0.1
# import pandas as pd
# data_df = pd.DataFrame(succ_rate)  # 利用pandas库对数据进行格式转换
# writer = pd.ExcelWriter('succ_rate.xlsx')  # 生成一个excel文件
# data_df.to_excel(writer, 'page_1')  # 数据写入excel文件
# writer.save()  # 保存excel文件

# index = [1.0,1.2,1.4,1.6,1.8,2.0]
index = [r'$\Omega_{}$'.format(i+1) for i in range(7)]+[r'AI$y$']
# import scipy.io
# scipy.io.savemat('succ_rate.mat', mdict={'data': succ_rate, })
# left = np.zeros(6)
# for i in range(7):
#     plt.plot([1.0,1.2,1.4,1.6,1.8,2.0],succ_rate[i],c=seven_color[6-i],marker=marker_list[i],label=r'$\Omega_{}$'.format(i+1),lw=2.0,markersize=8)
    # plt.barh(index, succ_rate[i], left=left, color=seven_color[i])
    # left = left + succ_rate[i]
fontsize = 18
labelsize= 15

succ_rate = np.load('succ_rate.npy')
seven_color = [[75/255,102/255,173/255],[98/255,190/255,166/255],
               [205/255,234/255,157/255],'gold',
               [253/255,186/255,107/255],[235/255,96/255,70/255],[163/255,6/255,67/255]
               ]
plt.bar(index,np.append(np.mean(succ_rate,axis=1),1.0),width=0.6,color=seven_color,alpha=0.6,lw=2.0,edgecolor=seven_color)
for i in range(7):
    plt.errorbar(index[i:i+1],np.mean(succ_rate,axis=1)[i],np.std(succ_rate,axis=1)[i],ecolor=seven_color[i],capsize=6)
    plt.scatter(np.random.uniform(-0.2+1.0*i, 0.2+1.0*i, 6), succ_rate[i], color=seven_color[i])
plt.yticks([0,1],['0','100%'],fontsize=labelsize)
plt.ylabel('Success Rate',fontsize=fontsize,labelpad=-45)
plt.xlabel('Type',fontsize=fontsize)
plt.xticks(fontsize=labelsize)

plt.show()





# data = torch.load('./data/data_root_AI_100_50.pt')[:,:5000]
# succ = success_rate(data)
# print(len(succ)/50)
# s = 12
# plt.plot(np.arange(data.shape[1]),data[s,:,2])
# plt.plot(np.arange(data.shape[1]),data[s,:,3])
# plt.show()


# for j in range(7):
#     data = torch.load('./data/data_para_{}_AI_1.4_50.pt'.format(j+1))
#     print(success_rate(data[:,:10000,:],1000,1.0))

# data = torch.load('./data/data_para_1_AI_1.0_50.pt')
# print(success_rate(data[:,:10000,:]))
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     s = i + 9*1
#     plt.plot(np.arange(data.shape[1]),data[s,:,0])
#     plt.plot(np.arange(data.shape[1]),data[s,:,1])
# plt.show()

