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

    def __init__(self, n_input, n_hidden, n_output,u_max):
        super(ControlNet, self).__init__()
        self.layer1 = SpectralNorm(torch.nn.Linear(n_input, n_hidden))
        # self.layer2 = SpectralNorm(torch.nn.Linear(n_hidden, n_hidden))
        self.layer3 = SpectralNorm(torch.nn.Linear(n_hidden, n_output))
        self.upperbound = u_max
        self.lowerbound = -self.upperbound

    def function(self, x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        # h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_1)
        return out

    def bound(self, x):
        x1 = torch.where(x < self.upperbound, x, 0.)
        return torch.where(self.lowerbound < x, x1, 0.)

    def forward(self, x):
        target = torch.zeros_like(x)
        u = self.function(x)
        return self.bound(u*x)

class MultiControlNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output,index,dim,u_max):
        '''
        :param n_input:
        :param n_hidden:
        :param n_output:
        :param index: outdegree排前top-k的节点在连接矩阵A中的位置，即按列求和之后向量中的位置坐标，以列表传入
        '''
        super(MultiControlNet, self).__init__()
        self.num = len(index)
        self.index = index
        self.dim = dim
        self.controller = ControlNet(n_input,n_hidden,n_output,u_max)
        # self.controlnets = []
        # for i in range(self.num):
        #     self.controlnets.append(ControlNet(n_input,n_hidden,n_output))

    def forward(self, x):
        output = self.controller(x)
        mask = torch.zeros_like(x)
        for i in range(self.num):
            ind = self.index[i]
            mask[:,ind],mask[:,ind+self.dim] = 1.0,1.0
        return output*mask

    # def forward(self, x):
    #     output = torch.zeros_like(x)
    #     for i in range(self.num):
    #         ind = self.index[i]
    #         controller = self.controlnets[i](torch.cat((x[:,ind:ind+1],x[:,ind+self.dim:ind+self.dim+1]),dim=1))
    #         output[:,ind],output[:,ind+self.dim] = controller[:,0],controller[:,1]
    #     return output


class SingleControlNet(torch.nn.Module):

    def __init__(self, index,dim,u_max):
        '''
        :param n_input:
        :param n_hidden:
        :param n_output:
        :param index: outdegree排前top-k的节点在连接矩阵A中的位置，即按列求和之后向量中的位置坐标，以列表传入
        '''
        super(SingleControlNet, self).__init__()
        self.num = len(index)
        self.index = index
        self.dim = dim
        self.controller = ControlNet(2,6,2,u_max)

    def forward(self, x):
        output = torch.zeros_like(x)
        for i in range(self.num):
            ind = self.index[i]
            ind_x = torch.cat((x[:,ind:ind+1],x[:,ind+self.dim:ind+self.dim+1]),dim=1)
            u = self.controller(ind_x)
            output[:,ind],output[:,ind+self.dim] = u[:,0],u[:,1]
        return output

class Model_FN(nn.Module):

    def __init__(self,index,n_input, n_hidden, n_output,dim,A,G_max,u_max,input_shape,layer_sizes=[64, 64,1],smooth_relu_thresh=0.1, eps=1e-3):
        super(Model_FN, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.a = torch.tensor([0.7]).to(self.device)
        self.b = torch.tensor([0.8]).to(self.device)
        self.epi = torch.tensor([0.2]).to(self.device)
        self.tau = torch.tensor([0.1]).to(self.device)
        self.I = torch.tensor([1.0]).to(self.device)

        self.W = torch.ones_like(A, requires_grad=True).to(self.device)
        self.edge_index = torch.nonzero(A).T  # 这里需要转置一下
        self.edge_index = self.edge_index.to(self.device)
        self.weight = torch.ones([self.edge_index.shape[1]]).to(self.device)
        self.weight.requires_grad = True

        self.mask = A  # 连接结构
        # self.G_max = G_max
        self.u_max = u_max
        self.index = index
        self.mask = self.mask.to(self.device)
        self.G_max = torch.tensor([G_max])
        self.G_max = self.G_max.to(self.device)
        # self.u_max = torch.tensor([u_max]).to(self.device)


        np.random.seed(10)
        # self.A = self.mask * torch.from_numpy(np.random.normal(1.0, 0.1, [dim, dim])) / (dim - 1)
        self.A = self.mask / (dim - 1)
        self.strength = torch.tensor([0.001]).to(self.device)
        self.scale = torch.tensor([1.0]).to(self.device)
        self.dim = dim
        self._eps = eps
        self.input_shape = input_shape
        self._control = ControlNet(n_input,n_hidden,n_output,u_max).to(self.device)
        # self._control = MultiControlNet(n_input, n_hidden, n_output,index,dim,u_max) #np.random.choice(np.arange(dim, dtype=np.int64),30,replace=False)
        # self._control = SingleControlNet(index,dim,u_max).to(self.device) #np.random.choice(np.arange(dim, dtype=np.int64),30,replace=False)
        self._lya = ICNN(input_shape, layer_sizes, smooth_relu_thresh,eps).to(self.device)


    def index2pos(self, x):
        mask = torch.zeros_like(x)
        for i in range(len(self.index)):
            ind = self.index[i]
            mask[:,ind],mask[:,ind+self.dim] = 1.0,1.0
        return (x*mask).to(self.device)

    def laplace(self,A):
        diag_L = torch.sum(A,dim=1)
        L = torch.diag(diag_L)-A
        L = L.to(self.device)
        return -L

    def H(self,u):
        # return 1/(1+torch.exp(-u/self.tau))
        return u

    def constraint(self,W):
        return torch.tanh(W)*self.G_max[0]
        # W = W * self.G_max[0]
        # return W.to(self.device)
        # W1 = torch.where(W < self.G_max, W, 0.)
        # W2 = torch.where(-self.G_max < W, W1, 0.)
        # return W2


    def FN(self,t,state):
        dx = torch.zeros_like(state)
        L = self.dim
        u,v = state[:,0:L],state[:,L:2*L]
        dx[:, 0*L:1*L] = u-u**3/3-v+self.I+self.strength*torch.mm(self.laplace(self.A),self.H(u).T).T
        dx[:, 1*L:2*L] = self.epi*(u+self.a-self.b*v)#+self.strength*torch.mm(self.laplace(self.A),self.H(v).T).T
        return dx

    def lyapunov_exp(self,strength):
        jacob = torch.kron(torch.eye(self.dim), torch.tensor([[1., -1.], [self.epi, -self.epi * self.b]])) + \
                strength * torch.kron(self.laplace(self.mask), torch.tensor(
            [[1 / self.tau , 0.0], [0.0, 0.0]]))  # -1/self.tau*math.exp(-2.0/self.tau)
        return torch.linalg.eig(jacob)

    def Jacob_FN(self,t,state):
        jacob = torch.kron(torch.eye(self.dim).to(self.device), torch.tensor([[1., -1.], [self.epi, -self.epi * self.b]]).to(self.device)) + \
                   self.strength*torch.kron(self.laplace(self.A),torch.tensor([[1.0, 0.0], [0.0,0.0]]).to(self.device)) #-1/self.tau*math.exp(-2.0/self.tau)
        dx = torch.mm(jacob,state.T).T
        return dx

    def Jacob_FN_g1(self,t,state):
        '''
        topology control
        '''
        # W = torch.sparse_coo_tensor(self.edge_index, torch.nn.functional.softmax(self.weight, dim=0),
        #                                  self.mask.shape).to_dense().to(self.device)
        # jacob = torch.kron(self.laplace(self.constraint(self.W)*self.mask),torch.tensor([[ 1, 0.0], [0.0,0.0]]).to(self.device)) #-1/self.tau*math.exp(-2.0/self.tau)
        jacob = torch.kron(self.laplace(self.constraint(self.W)*self.mask),torch.tensor([[1/self.tau, 0.0], [0.0,0.0]]).to(self.device)) #-1/self.tau*math.exp(-2.0/self.tau)
        dx = torch.mm(self.strength*jacob,state.T).T
        return dx

    def Jacob_FN_g2(self,t,state):
        '''
        external force
        '''
        u = self.index2pos(self._control(state)).to(self.device)
        return u

    def Jacob_FN_g3(self,t,state):
        '''
        external force on partial nodes
        '''
        dstate = torch.zeros_like(state)
        L = self.dim
        u = self._control(state[:,0:L])
        dstate[:,0:L] = u
        return dstate

    def FN_g1(self, t, state):
        # W = torch.sparse_coo_tensor(self.edge_index, torch.nn.functional.softmax(self.weight, dim=0),
        #                                  self.mask.shape).to_dense().to(self.device)
        dstate = torch.zeros_like(state)
        L = self.dim
        dstate[:, 0:L] = torch.mm(self.laplace(self.constraint(self.W)*self.mask),self.H(state[:, 0:L]).T).T
        # dstate[:, 0:L] = self.G_max*torch.mm(self.laplace(self.A),self.H(state[:, 0:L]).T).T
        return dstate.unsqueeze(2)

    def FN_g2(self, t, state):
        dstate = torch.zeros_like(state)
        st = torch.zeros_like(state)
        L = self.dim
        u,v = state[:,0:L],state[:,L:2*L]
        st[:, 0:L] = u-u[:,0:1].repeat(1,L)
        st[:, L:2*L] = v-v[:,0:1].repeat(1,L)
        # st[:, 0:L] = u-u[:,0:L].mean(dim=1).unsqueeze(1).repeat(1,L)
        # st[:, L:2*L] = v-v[:,0:L].mean(dim=1).unsqueeze(1).repeat(1,L)
        dstate[0,0:2*L] = self.index2pos(self._control(st)).to(self.device)
        return dstate.unsqueeze(2)

    def FN_g3(self, t, state):
        dstate = torch.zeros_like(state)
        st = torch.zeros_like(state)
        L = self.dim
        u,v = state[:,0:L],state[:,L:2*L]
        st[:, 0:L] = u-u[:,0:1].repeat(1,L)
        st[:, L:2*L] = v-v[:,0:1].repeat(1,L)
        # st[:, 0:L] = u-u[:,0:L].mean(dim=1).unsqueeze(1).repeat(1,L)
        # st[:, L:2*L] = v-v[:,0:L].mean(dim=1).unsqueeze(1).repeat(1,L)
        dstate[0,0:L] = self._control(st[:,0:L])[:,0:L]
        return dstate.unsqueeze(2)

    def FN_linear_g(self, t, state):
        dstate = torch.zeros_like(state)
        st = torch.zeros_like(state)
        L = self.dim
        u,v = state[:,0:L],state[:,L:2*L]
        st[:, 0:L] = u-u[:,1:2].repeat(1,L)
        st[:, L:2*L] = v-v[:,1:2].repeat(1,L)
        # st[:, 0:L] = u-u[:,0:L].mean(dim=1).unsqueeze(1).repeat(1,L)
        # st[:, L:2*L] = v-v[:,0:L].mean(dim=1).unsqueeze(1).repeat(1,L)
        dstate[0,0:2*L] = self.index2pos(3.0*st)
        return dstate.unsqueeze(2)


    def train_g(self, t, state):
        dstate = torch.zeros_like(state)
        L = self.dim
        m = len(state)
        u,v = state[:,0:L],state[:,L:2*L]
        # u_s, v_s = self.R[0:m,0],self.R[0:m,1]
        '''
        distributed control mechanism
        '''
        # u = self._control(self.H(state[:,0:L]))

        # u = self._control(torch.cat((a,b,c),dim=1))
        dstate[:, 0:L] = torch.mm(self.laplace(self.mask*self.W),self.H(u).T).T
        # dstate[:, 0:L] = u
        # return u.unsqueeze(2)
        # return -50*state.unsqueeze(2)
        return dstate.unsqueeze(2)

    def FN_g(self, t, state):
        dstate = torch.zeros_like(state)
        st = torch.zeros_like(state)
        L = self.dim
        u,v = state[:,0:L],state[:,L:2*L]
        st[:, 0:L] = u-u[:,0:1].repeat(1,L)
        st[:, L:2*L] = v-v[:,0:1].repeat(1,L)
        # u = self._control(self.H(st[:,0:L]))
        # dstate[:, 0:L] = u
        dstate[:, 0:L] = self.strength*torch.mm(self.laplace(self.constraint(self.W)*self.mask),self.H(state[:, 0:L]).T).T
        # u = self._control(torch.cat((a,b,c),dim=1))
        # dstate[:,0:L] = 3*st[:,0:L]
        # return u.unsqueeze(2)
        return dstate.unsqueeze(2)


class Model_FN_cpu(nn.Module):

    def __init__(self,index,n_input, n_hidden, n_output,dim,A,G_max,u_max,input_shape,layer_sizes=[64, 64,1],smooth_relu_thresh=0.1, eps=1e-3):
        super(Model_FN_cpu, self).__init__()

        self.a = 0.7
        self.b = 0.8
        self.epi = 0.2
        self.tau = 0.1
        self.I = 1.0



        self.W = torch.ones_like(A,requires_grad = True)
        self.mask = A  # 连接结构
        self.G_max = G_max
        self.u_max = u_max
        self.index = index
        self.mask = self.mask

        np.random.seed(10)
        # self.A = self.mask * torch.from_numpy(np.random.normal(1.0, 0.1, [dim, dim])) / (dim - 1)
        self.A = self.mask / (dim - 1)
        self.strength = 0.05
        self.scale = 1.0
        self.dim = dim
        self._eps = eps
        self.input_shape = input_shape
        self._control = ControlNet(n_input,n_hidden,n_output,u_max)
        # self._control = MultiControlNet(n_input, n_hidden, n_output,index,dim,u_max) #np.random.choice(np.arange(dim, dtype=np.int64),30,replace=False)
        # self._control = SingleControlNet(index,dim,u_max) #np.random.choice(np.arange(dim, dtype=np.int64),30,replace=False)
        self._lya = ICNN(input_shape, layer_sizes, smooth_relu_thresh,eps)


    def index2pos(self, x):
        mask = torch.zeros_like(x)
        for i in range(len(self.index)):
            ind = self.index[i]
            mask[:,ind],mask[:,ind+self.dim] = 1.0,1.0
        return x*mask

    def laplace(self,A):
        diag_L = torch.sum(A,dim=1)
        L = torch.diag(diag_L)-A
        L = L
        return -L

    def H(self,u):
        # return 1/(1+torch.exp(-u/self.tau))
        return u

    def constraint(self,W):
        return torch.tanh(W)*self.G_max/(self.dim - 1)
        # W = W * self.G_max[0]
        # return W
        # W1 = torch.where(W < self.G_max, W, 0.)
        # W2 = torch.where(-self.G_max < W, W1, 0.)
        # return W2


    def FN(self,t,state):
        dx = torch.zeros_like(state)
        L = self.dim
        u,v = state[:,0:L],state[:,L:2*L]
        dx[:, 0*L:1*L] = u-u**3/3-v+self.I+self.strength*torch.mm(self.laplace(self.A),self.H(u).T).T
        dx[:, 1*L:2*L] = self.epi*(u+self.a-self.b*v)#+self.strength*torch.mm(self.laplace(self.A),self.H(v).T).T
        return dx

    def lyapunov_exp(self,strength):
        jacob = torch.kron(torch.eye(self.dim), torch.tensor([[1., -1.], [self.epi, -self.epi * self.b]])) + \
                strength * torch.kron(self.laplace(self.mask), torch.tensor(
            [[1 / self.tau , 0.0], [0.0, 0.0]]))  # -1/self.tau*math.exp(-2.0/self.tau)
        return torch.linalg.eig(jacob)

    def Jacob_FN(self,t,state):
        jacob = torch.kron(torch.eye(self.dim), torch.tensor([[1., -1.], [self.epi, -self.epi * self.b]])) + \
                   self.strength*torch.kron(self.laplace(self.A),torch.tensor([[1, 0.0], [0.0,0.0]])) #1/self.tau
        dx = torch.mm(jacob,state.T).T
        return dx

    def Jacob_FN_g1(self,t,state):
        '''
        topology control
        '''
        jacob = torch.kron(self.laplace(self.constraint(self.W)*self.mask),torch.tensor([[1/self.tau, 0.0], [0.0,0.0]])) #-1/self.tau*math.exp(-2.0/self.tau)
        dx = self.strength*torch.mm(jacob,state.T).T
        return dx
        # return torch.zeros_like(dx)

    def Jacob_FN_g2(self,t,state):
        '''
        external force
        '''
        u = self.index2pos(self._control(state))
        return u

    def FN_g1(self, t, state):
        dstate = torch.zeros_like(state)
        L = self.dim
        dstate[:, 0:L] = self.strength*torch.mm(self.laplace(self.constraint(self.W)*self.mask),self.H(state[:, 0:L]).T).T
        return dstate.unsqueeze(2)

    def FN_g2(self, t, state):
        dstate = torch.zeros_like(state)
        st = torch.zeros_like(state)
        L = self.dim
        u,v = state[:,0:L],state[:,L:2*L]
        st[:, 0:L] = u-u[:,0:1].repeat(1,L)
        st[:, L:2*L] = v-v[:,0:1].repeat(1,L)
        # st[:, 0:L] = u-u[:,0:L].mean(dim=1).unsqueeze(1).repeat(1,L)
        # st[:, L:2*L] = v-v[:,0:L].mean(dim=1).unsqueeze(1).repeat(1,L)
        dstate[0,0:2*L] = self.index2pos(self._control(st))
        return dstate.unsqueeze(2)

    def FN_linear_g(self, t, state):
        dstate = torch.zeros_like(state)
        st = torch.zeros_like(state)
        L = self.dim
        u,v = state[:,0:L],state[:,L:2*L]
        st[:, 0:L] = u-u[:,1:2].repeat(1,L)
        st[:, L:2*L] = v-v[:,1:2].repeat(1,L)
        # st[:, 0:L] = u-u[:,0:L].mean(dim=1).unsqueeze(1).repeat(1,L)
        # st[:, L:2*L] = v-v[:,0:L].mean(dim=1).unsqueeze(1).repeat(1,L)
        dstate[0,0:2*L] = self.index2pos(5.0*st)
        return dstate.unsqueeze(2)







