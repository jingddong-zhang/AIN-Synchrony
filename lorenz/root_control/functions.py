import numpy as np
from numpy.core.defchararray import mod
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_normalization import SpectralNorm
from MonotonicNN import MonotonicNN
import math
import timeit
from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event
import torchsde
import networkx as nx
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
        self.upperbound = 100. # 100 for AI, 300 for linear
        self.lowerbound = -self.upperbound

    def function(self, x):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out

    def bound(self, x):
        relu = (x-self.lowerbound).relu()+self.lowerbound
        # relu = self.upperbound-(self.upperbound-relu).relu()
        return torch.where(relu < self.upperbound, relu, self.upperbound)

    def forward(self, x):
        target = torch.zeros_like(x)
        u = self.function(x)
        u0 = self.function(target)
        # x = x[:,0:self.dim*3]
        # return self.bound(u-u0)
        # return self.bound(u*x)
        return self.bound(u*(x[:,0:1]+x[:,1:2]+x[:,2:3]))

class ControlNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output,dim):
        super(ControlNet, self).__init__()
        self.layer1 = SpectralNorm(torch.nn.Linear(n_input, n_hidden))
        self.layer2 = SpectralNorm(torch.nn.Linear(n_hidden, n_hidden))
        self.layer3 = SpectralNorm(torch.nn.Linear(n_hidden, n_output))
        self.dim = dim
        self.upperbound = 100. # 100 for AI, 300 for linear
        self.lowerbound = -self.upperbound

    def function(self, x):
        sigmoid = torch.nn.ReLU()
        h_1 = sigmoid(self.layer1(x))
        h_2 = sigmoid(self.layer2(h_1))
        out = self.layer3(h_2)
        return out

    def bound(self, x):
        relu = (x-self.lowerbound).relu()+self.lowerbound
        # relu = self.upperbound-(self.upperbound-relu).relu()
        return torch.where(relu < self.upperbound, relu, self.upperbound)

    def forward(self, x):
        target = torch.zeros_like(x)
        u = self.function(x)
        u0 = self.function(target)
        # x = x[:,0:self.dim*3]
        # return self.bound(u-u0)
        # return self.bound(u*x)
        return self.bound(u*(x[:,0:1]+x[:,1:2]+x[:,2:3]))

class Model_lorenz(nn.Module):

    def __init__(self,n_input, n_hidden, n_output,dim,R,input_shape,layer_sizes=[64, 64],smooth_relu_thresh=0.1, eps=1e-3):
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
        self.delta_b = torch.tensor(0.5, requires_grad=True)

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

    def control_lorenz(self,t,state):
        dstate = torch.zeros_like(state)
        L = self.dim
        x,y,z = state[:,0:L],state[:,L:2*L],state[:,2*L:3*L]
        state[:, 0] = x[:,1]-x[:,0]
        state[:, 1] = y[:,1]-y[:,0]
        state[:, 2] = z[:,1]-z[:,0]
        u = self._control(state)[:,0]
        dstate[:, 0*L:1*L] = self.sigma*(y-x)
        dstate[:, 1*L:2*L] = self.rho*x-y-x*z
        dstate[:, 2*L:3*L] = x*y-self.beta*z
        dstate[:,3] += u*(y[:,1]-x[:,1])
        return dstate

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
        # u = self._control(torch.cat((state,self.R),dim=1))
        # mask = torch.tensor([[1.,1.,1.,0.,0.,0.,0.]]).repeat(len(u),L)
        # u *= mask

        # u = self._control(torch.cat((a,b,c),dim=1))
        # dstate[:, 0] = u[:,0]#*(y-x+y_s-x_s)
        dstate[:, 1] = u[:, 1] #* (x + x_s)
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
        u = self._control(st)

        # u = self._control(torch.cat((a,b,c),dim=1))
        # dstate[:, 3] = u[:, 0]#*(y[:,1]-x[:,1])
        dstate[:, 4] = u[:, 1] #* x[:, 1]
        # dstate[:, 5] = u[:, 2]
        # dstate[:, 4] = self._control.bound(u[:, 1]*x[:,1])
        # dstate[:, 5] = self._control.bound(u[:, 2]*z[:,1])
        # dstate[:,4] = self._control.bound(5*(st[:,1]+st[:,0]+st[:,2]))# 根节点为y变量，对根节点进行控制
        # dstate[:, 3:6] = self._control.bound(10*st)
        # return u.unsqueeze(2)
        # return -50*state.unsqueeze(2)
        return dstate.unsqueeze(2)

    def linear_g(self, t, state):
        dstate = torch.zeros_like(state)
        st = torch.zeros([len(state),3])
        L = self.dim
        m = len(state)
        x,y,z = state[:,0:L],state[:,L:2*L],state[:,2*L:3*L]
        st[:, 0] = x[:,1]-x[:,0]
        st[:, 1] = y[:,1]-y[:,0]
        st[:, 2] = z[:,1]-z[:,0]
        k = 2.
        dstate[:,4] = self._control.bound(k*(st[:,1]+st[:,0]+st[:,2]))# 根节点为y变量，对根节点进行控制
        return dstate.unsqueeze(2)


class Model_lorenz_linear(nn.Module):

    def __init__(self, k):
        super(Model_lorenz_linear, self).__init__()

        self.sigma = 10.
        self.rho = 28.
        self.beta = 8 / 3
        self.k = k
        self.upperbound = 100. # 300对应数据data_root_linear
        self.lowerbound = -self.upperbound
        self.scale = 1.
        self.noise_type = "scalar"
        self.sde_type = "ito"



    def bound(self, x):
        relu = (x-self.lowerbound).relu()+self.lowerbound
        # relu = self.upperbound-(self.upperbound-relu).relu()
        return torch.where(relu < self.upperbound, relu, self.upperbound)

    def f(self, t, state):
        dx = torch.zeros_like(state)
        L = 2
        x, y, z = state[:, 0:L], state[:, L:2 * L], state[:, 2 * L:3 * L]
        '''
        Lorenz: drive-response
        '''
        # 向量场赋值
        dx[:, 0 * L:1 * L] = self.sigma * (y - x)
        dx[:, 1 * L:2 * L] = self.rho * x - y - x * z * self.scale
        dx[:, 2 * L:3 * L] = x * y * self.scale - self.beta * z
        return dx

    def f_control(self, t, state):
        dx = torch.zeros_like(state)
        L = 2
        x, y, z = state[:, 0:L], state[:, L:2 * L], state[:, 2 * L:3 * L]
        '''
        Lorenz: drive-response
        '''
        # 向量场赋值
        dx[:, 0 * L:1 * L] = self.sigma * (y - x)
        dx[:, 1 * L:2 * L] = self.rho * x - y - x * z * self.scale
        dx[:, 2 * L:3 * L] = x * y * self.scale - self.beta * z
        st = torch.zeros([len(state), 3])
        st[:, 0] = x[:, 1] - x[:, 0]
        st[:, 1] = y[:, 1] - y[:, 0]
        st[:, 2] = z[:, 1] - z[:, 0]

        dx[:, 4] += self.bound(self.k * (st[:, 1] + st[:, 0] + st[:, 2])) # 根节点为y变量，对根节点进行控制
        return dx

    def train_linear_g(self, t, state):
        dstate = torch.zeros_like(state)
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        dstate[:, 1] = self.bound(self.k*(x+y+z))
        return dstate.unsqueeze(2)

    def bound_g(self, t, state):
        dstate = torch.zeros_like(state)
        st = torch.zeros([len(state), 3])
        L = 2
        m = len(state)
        x, y, z = state[:, 0:L], state[:, L:2 * L], state[:, 2 * L:3 * L]
        st[:, 0] = x[:, 1] - x[:, 0]
        st[:, 1] = y[:, 1] - y[:, 0]
        st[:, 2] = z[:, 1] - z[:, 0]

        dstate[:, 4] = self.bound(self.k * (st[:, 1] + st[:, 0] + st[:, 2]))  # 根节点为y变量，对根节点进行控制
        return dstate.unsqueeze(2)

    def g(self, t, state):
        dstate = torch.zeros_like(state)
        st = torch.zeros([len(state), 3])
        L = 2
        m = len(state)
        x, y, z = state[:, 0:L], state[:, L:2 * L], state[:, 2 * L:3 * L]
        st[:, 0] = x[:, 1] - x[:, 0]
        st[:, 1] = y[:, 1] - y[:, 0]
        st[:, 2] = z[:, 1] - z[:, 0]

        dstate[:, 4] = self.k * (st[:, 1] + st[:, 0] + st[:, 2]) # 根节点为y变量，对根节点进行控制
        return dstate.unsqueeze(2)

    def non_g(self, t, state):
        dstate = torch.zeros_like(state)
        return dstate.unsqueeze(2)



def generate(m,model,true_y0,g_case='lorenz_g',save=False):
    N = 5000
    # m = 50
    t = torch.linspace(0., 10., N)  # 时间点
    data = torch.zeros([m, 2 * N, 6])
    for i in range(m):
        seed = i+1
        torch.manual_seed(seed)
        # np.random.seed(seed)
        noise = torch.Tensor(1, 3).uniform_(-2.5, 2.5)
        true_y0[:, 0:6:2] += noise
        with torch.no_grad():
            cont_y = torchsde.sdeint(model, true_y0, t, method='euler',
                                     names={'drift': 'lorenz', 'diffusion':g_case})[:,
                     0, :]
            data[i, :N, :] = cont_y
            cont_y = torchsde.sdeint(model, cont_y[-2:-1, :], t, method='euler',
                                     names={'drift': 'lorenz', 'diffusion': g_case})[:,
                     0, :]
            data[i, N:, :] = cont_y
        print('current iteration:{}'.format(i))
    if save:
        torch.save(data, './data/data_root_AI_100_{}.pt'.format(m))
        print('Successfully saved {} data'.format(m))
    return data

def generate_linear(true_y0):
    N = 5000
    m = 50
    K = 15
    t = torch.linspace(0., 10., N)  # 时间点
    data = torch.zeros([K,m, 2 * N, 6])
    for j in range(K):
        k = (j+1)
        for i in range(m):
            model = Model_lorenz_linear(k)
            seed = i + 1
            torch.manual_seed(seed)
            # np.random.seed(seed)
            noise = torch.Tensor(1, 3).uniform_(-2.5, 2.5)
            true_y0[:, 0:6:2] += noise
            with torch.no_grad():
                cont_y = torchsde.sdeint(model, true_y0, t, method='euler',
                                         names={'drift': 'f', 'diffusion':'bound_g'})[:,
                         0, :]
                data[j,i, :N, :] = cont_y
                cont_y = torchsde.sdeint(model, cont_y[-2:-1, :], t, method='euler',
                                         names={'drift': 'f', 'diffusion': 'bound_g'})[:,
                         0, :]
                data[j,i, N:, :] = cont_y
            print('current iteration:{},{}'.format(j,i))
    torch.save(data, './data/data_root_linear_100_{}_{}_v0.pt'.format(K,m))

def generate_linear_ode(true_y0):
    N = 5000
    m = 10
    K = 10
    t = torch.linspace(0., 10., N)  # 时间点
    data = torch.zeros([K,m, 2 * N, 6])
    for j in range(K):
        k = j+1
        for i in range(m):
            model = Model_lorenz_linear(k)
            seed = i + 1
            torch.manual_seed(seed)
            np.random.seed(seed)
            noise = torch.Tensor(1, 3).uniform_(-2.5, 2.5)
            true_y0[:, 0:6:2] += noise
            with torch.no_grad():
                cont_y = torchsde.sdeint(model, true_y0, t, method='euler',
                                         names={'drift': 'f_control', 'diffusion':'non_g'})[:,
                         0, :]
                data[j,i, :N, :] = cont_y
                cont_y = torchsde.sdeint(model, cont_y[-2:-1, :], t, method='euler',
                                         names={'drift': 'f_control', 'diffusion': 'non_g'})[:,
                         0, :]
                data[j,i, N:, :] = cont_y
            print('current iteration:{},{}'.format(j,i))
    torch.save(data, './data/data_root_linear_ode_100.pt')

def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
    r'''
    Compute the gradient of `outputs` with respect to `inputs`
    ```
    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    ```
    '''
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])


def jacobian(outputs, inputs, create_graph=False):
    r'''
    Compute the Jacobian of `outputs` with respect to `inputs`
    ```
    jacobian(x, x)
    jacobian(x * y, [x, y])
    jacobian([x * y, x.sqrt()], [x, y])
    ```
    '''
    if torch.is_tensor(outputs):
        outputs = [outputs]
    else:
        outputs = list(outputs)

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    jac = []
    for output in outputs:
        output_flat = output.view(-1)
        output_grad = torch.zeros_like(output_flat)
        for i in range(len(output_flat)):
            output_grad[i] = 1
            jac += [gradient(output_flat, inputs, output_grad, True, create_graph)]
            output_grad[i] = 0
    return torch.stack(jac)

def hessian(output, inputs, out=None, allow_unused=False, create_graph=False):
    r'''
    Compute the Hessian of `output` with respect to `inputs`
    ```
    hessian((x * y).sum(), [x, y])
    ```
    '''
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    numel = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(numel, numel)

    row_index = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(output, inp, create_graph=True, allow_unused=allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = gradient(grad[j], inputs[i:], retain_graph=True, create_graph=create_graph)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[row_index, row_index:].add_(row.type_as(out))  # row_index's row
            if row_index + 1 < numel:
                out[row_index + 1:, row_index].add_(row[1:].type_as(out))  # row_index's column
            del row
            row_index += 1
        del grad

    return out



def convexity():
    V_model = ICNN((3,), [18, 18, 1], 0.1, 1e-3)  # D_in = 3 , H1 = 3*D_in
    # V_model2 = ICNN((3,), [18, 18, 1], 0.1, 1e-3)  # D_in = 3 , H1 = 3*D_in

    torch.manual_seed(369)
    n = 20
    x,y,z = torch.linspace(-2.5,2.5,n),torch.linspace(-2.5,2.5,n),torch.linspace(-2.5,2.5,n)
    X,Y,Z = torch.meshgrid(x,y,z)
    data = torch.stack([X, Y,Z], dim=3).view(-1,3).requires_grad_(True)
    # data = torch.Tensor(n,3).uniform_(-2.5,2.5).requires_grad_(True)
    eigens = torch.zeros([16,len(data)])
    for k in range(15):
        V_model.load_state_dict(torch.load('./data/V_linear_{}_100.pkl'.format(k + 1)))
        for i in range(len(data)):
            x = data[i:i+1,:]
            hessV = hessian(V_model(x).sum(),x)
            # eigens[k,i] =  torch.real(torch.linalg.eigvals(hessV)).max()
            eigens[k, i] = torch.real(torch.linalg.eigvals(hessV).mean())
        print(k)
    V_model.load_state_dict(torch.load('./data/V_AI_2.5.pkl'))
    for i in range(len(data)):
        x = data[i:i + 1, :]
        hessV = hessian(V_model(x).sum(), x)
        # eigens[-1, i] = torch.real(torch.linalg.eigvals(hessV)).max()
        eigens[-1, i] = torch.real(torch.linalg.eigvals(hessV).mean())
    torch.save(eigens,'./data/eigvalstrace_{}_v1.pt'.format(n))


def convexity_v1():
    V_hess = torch.load('./data/V_hess_root_2000.pt')
    epoch,N = V_hess.shape[0], V_hess.shape[1]
    eigens = torch.zeros([21, N])
    for i in range(epoch):
        for j in range(N):
            eigens[i,j] = torch.real(torch.linalg.eigvals(V_hess[i,j,:,j,:]).mean())
        print('current epoch:',i)
    torch.save(eigens, './data/eigvalsmean_AI_root_training_2000.pt')

# convexity_v1()
# convexity()
# for i in range(10):
#     data = torch.load('./data/loss_{}_100.pt'.format(i+1))
#     print(data.min())
