import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch
from functions import *
import scipy.io as scio
import networkx as nx
import os.path as osp
import os

torch.set_default_dtype(torch.float64)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.deterministic = True
print(device)

def success_rate(data, T=15000, delta=0.1):
    data = data[T:]
    L = int(data.shape[1] / 2)
    secc_num = 0
    x1 = data[:, 0]
    for i in range(1, L):
        x = data[:, i]
        diff = np.abs(x1 - x)
        if np.mean(diff) < delta:
            secc_num += 1
    order = secc_num / (L - 1)
    return order


def pnas_order(data, T=15000, delta=0.1):
    L = int(data.shape[1] / 2)
    data = data[T:, 0:L]
    mean1 = np.mean(data, axis=1)
    std1 = np.std(mean1)
    std2 = np.mean(np.std(data, axis=0))
    order = std1 / std2
    return order

def trans_P(dim):
    P = torch.zeros([dim,dim])
    for i in range(1,dim):
        v = torch.zeros([dim])
        for j in range(i):
            v[j] = 1/math.sqrt(i*(i+1))
        v[i] = -(i)/math.sqrt(i*(i+1))
        P[i,:] = v
    P = P.to(device)
    return P

def dom_laplace(dim):
    k = 1.0
    L = torch.zeros([dim,dim])
    for i in range(1,dim):
        v = torch.zeros([1,dim])
        for j in range(i):
            v[0,j] = 1/math.sqrt(i*(i+1))
        v[0,i] = -(i)/math.sqrt(i*(i+1))
        new_v = (1+2*k)*torch.mm(v.T,v)
        L += new_v
    L = L.to(device)
    return L

P = trans_P(11)
L = dom_laplace(11)
# print(A)
dim = len(L)




data_size = 2000
N = 2000             # sample size
D_in = 2*dim            # input dimension
H1 = 3*D_in             # hidden dimension
D_out = 2*dim           # output dimension

start = timeit.default_timer()
out_iters = 0
while out_iters < 1:
    # break
    setup_seed(369)
    data = torch.Tensor(data_size, D_in).uniform_(-4.7, 4.7).requires_grad_(True).to(device)

    u_max = 5.0 # 0.5,1.0,2.0
    model = Model_FN(D_in,H1,D_out,dim,L,P,u_max,(D_in,),[H1,1]).to(device)
    i = 0
    max_iters = 500
    learning_rate = 0.01

    optimizer = torch.optim.Adam([i for i in model.parameters()]+[model.W], lr=learning_rate)
    Loss = []
    r_f = torch.randn(D_in).to(device)  # hutch estimator vector


    while i < max_iters:
        # break
        s = torch.from_numpy(np.random.choice(np.arange(data_size, dtype=np.int64), N, replace=False))
        x = data[s].requires_grad_(True)
        f = model.Jacob_FN(0.0,x)
        g = model.Jacob_FN_g(0.0,x)

        V = torch.sum(x**2,dim=1)
        # V = model._lya(x)
        Vx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        r_Vxx = torch.autograd.grad(torch.sum(Vx * r_f, dim=1).sum(), x, create_graph=True)[0]

        L_V = torch.sum(Vx * f, dim=1) + 0.5 * torch.sum((torch.sum(g * r_f, dim=1).view(-1, 1) * g) * r_Vxx, dim=1)
        Vxg = torch.sum(Vx*g,dim=1)
        loss = -Vxg ** 2 / (V ** 2) + 2.5 * L_V / V
        loss = (F.relu(loss)).mean()

        Loss.append(loss.item())
        if loss.item() == min(Loss):
            best_control_model = model._control.state_dict()
            best_V_model = model._lya.state_dict()
        print(out_iters,i, 'total loss=',loss.item(),"Lyapunov Risk=",loss.item(),model.W.sum()) #,model._control.controlnets[0].layer1.weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        i += 1

    middle = timeit.default_timer()
    print('training time: {:.5f}'.format(middle - start))

    '''
    save model 
    '''
    # model._control.load_state_dict(best_control_model)

    '''
    load model
    '''
    # model._control.load_state_dict(torch.load('./data/AIcontrol_{}_50_{}_2.5_{}.pkl'.format(mode_list[out_iters],N,model._control.upperbound)))
    # generate(model,true_y0,mode_list[out_iters])

    def originaL(true_y0, N, dt, model):
        sol = torch.zeros([N, dim * 2])
        sol[0] = true_y0
        for i in range(N - 1):
            y = sol[i:i + 1]
            new_y = y + dt * model.FN(0.0, y)
            sol[i + 1:i + 2] = new_y
        return sol

    def controllerd(true_y0,N, dt, seed, model,mode):
        setup_seed(seed)
        # np.random.seed(seed)
        # N = 20000  # len(driving)
        sol,csol = torch.zeros([N, dim*2]).to(device),torch.zeros([N, dim*2]).to(device)
        sol[0],csol[0] = true_y0,true_y0
        if mode == 0:
            w = np.random.normal(0, 1, N) # common noise
        if mode == 1:
            w = torch.from_numpy(np.random.normal(0, 1, [N,dim])).repeat(1,2).to(device) # uncorrelated noise
        t = 0
        for i in range(N - 1):
            t += i*dt
            x = csol[i:i + 1]
            with torch.no_grad():
                u = model.FN_g(0.0, x)[:, :, 0]
            # u = model.FN_linear_g(0.0, x)[:, :, 0]
            new_x = x + dt * model.FN(t,x) + w[i] * np.sqrt(dt) * u
            # new_x = x + dt * (model.FN(0.0, x)-model.strength*u)
            csol[i + 1:i + 2] = new_x
            y = sol[i:i + 1]
            new_y = y + dt * model.FN(t,y)
            sol[i + 1:i + 2] = new_y
        return sol,csol

    m = 10
    length = 12000
    dt = 1e-2
    data_cont,data_true = np.zeros([m,length,dim*2]),np.zeros([m,length,dim*2])

    for mode in [0]:
        for k in range(m):
            true_y0 = torch.randn([1, 2 * dim]).to(device)  # 初值
            seed = k
            true_y,cont_y = controllerd(true_y0,length,dt,seed,model,mode)
            cont_y = cont_y.cpu().detach().numpy()
            true_y = true_y.cpu().detach().numpy()
            data_cont[k],data_true[k] = cont_y,true_y
            print(f'current step:{k}')
        np.save('./data/u_max={} m={}'.format(u_max,m),{'cont': data_cont, 'true': data_true})
    L = dim






    plt.subplot(131)
    T = 2000
    # for i in range(10):
    #     plt.plot(np.arange(T), cont_y[-T:, i],label='{}'.format(i))
    plt.imshow(cont_y[-10000:, 0:L].T, extent=[0, 500, 0, L], cmap='RdBu', aspect='auto')
    # plt.title(r'Controlled R={:.3f}'.format(np.std(cont_y[-2000:,:dim],axis=1).mean()))
    plt.title(r'Controlled R={:.3f}'.format(pnas_order(cont_y)))


    plt.subplot(132)
    plt.plot(np.arange(len(cont_y)), cont_y[:, 1]-cont_y[:,0])
    plt.title(r'$x_2-x_1$')

    # plt.subplot(132)
    # T=10000
    # plt.plot(np.arange(T),true_y[-T:,0]-true_y[-T:, 1])
    # plt.title(f'{np.max(true_y[:,:L]-np.repeat(true_y[:,0:1],L,axis=1))}')

    plt.subplot(133)
    # for i in range(10):
    #     plt.plot(np.arange(len(true_y)), true_y[:, i],label='{}'.format(i))
    plt.imshow(true_y[-10000:, 0:L].T, extent=[0, 500, 0, L], cmap='RdBu', aspect='auto')
    # plt.title('Original R={:.3f}'.format(np.std(true_y[-2000:,:dim],axis=1).mean()))
    plt.title('Original R={:.3f}'.format(pnas_order(true_y)))




    out_iters+=1
end = timeit.default_timer()


print('total time: {:.5f}'.format(end-start))
plt.show()