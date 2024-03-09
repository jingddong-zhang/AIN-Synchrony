import matplotlib.pyplot as plt
import numpy as np
import torch
from functions import *
import scipy.io as scio
import networkx as nx
from FVS import find_FVS
torch.set_default_dtype(torch.float64)




A = np.load('./data/topology/39 117 Chesapeake.npy')
# A = np.load('./data/topology/39 Di_ER k=4 seed=369.npy')

# A = torch.from_numpy(np.load('./data/topology/128 2137 baydry.npy')).to(torch.float64)
# A = torch.from_numpy(np.load('./data/topology/PPI1.npy')).to(torch.float64)

# A = nx.to_numpy_matrix(nx.star_graph(39))


dim = len(A)

indegree = A.sum(axis=1)
mode1 = np.where(indegree==0)[0]
FVS = np.array(find_FVS(A))
ind_1 = np.where(indegree==1)[0]
mode2 = np.unique(np.concatenate((mode1,FVS))) if len(FVS)>0 else np.unique(np.concatenate((mode1,ind_1)))
ind_2 = np.where(indegree<=np.ceil(indegree.mean()))[0]
mode3 = np.unique(np.concatenate((mode2,ind_2)))
mode4 = range(dim)
# print(len(mode2))

A = torch.from_numpy(A).to(torch.float64)
# n = 39
# A = torch.ones([n,n])-torch.eye(n)
data_size = 1000
N = 1000             # sample size
D_in = 2*dim            # input dimension
H1 = 3*D_in             # hidden dimension
D_out = 2*dim           # output dimension



# true_y0 = torch.randn([1, 2*dim])  # 初值
# true_y0 = torch.from_numpy(np.random.normal(1,0.1,[1,dim*2]))
#
# t = torch.linspace(0., 50., 1000+N)  # 时间点
# model = Model_FN([0],D_in,H1,D_out,dim,A,0,(D_in,),[10,10,1])
# with torch.no_grad():
#     true_y = odeint(model.FN, true_y0, t, method='dopri5')[:,0,:]
#
#
# for i in range(dim):
#     plt.plot(np.arange(N),true_y[-N:,i])
# plt.show()

start = timeit.default_timer()
out_iters = 0
while out_iters < 1:
    # break
    setup_seed(369)
    data = torch.Tensor(data_size, D_in).uniform_(-4.7, 4.7).requires_grad_(True)
    G_max = 10.0
    u_max = 2.0
    model = Model_FN_cpu(mode4,D_in,H1,D_out,dim,A,G_max,u_max,(D_in,),[H1,1])
    i = 0
    max_iters = 500
    learning_rate = 0.01

    optimizer = torch.optim.Adam([i for i in model.parameters()]+[model.W], lr=learning_rate,weight_decay=0.0)
    Loss = []
    r_f = torch.randn(D_in)  # hutch estimator vector

    # eig = np.real(model.lyapunov_exp(10.0)[0])
    # print(eig)

    while i < max_iters:
        # break
        s = torch.from_numpy(np.random.choice(np.arange(data_size, dtype=np.int64), N, replace=False))
        x = data[s].requires_grad_(True)
        f = model.Jacob_FN(0.0,x)
        g = model.Jacob_FN_g2(0.0,x)
        # x = x[:,0:3*dim]

        V = model._lya(x)
        Vx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        r_Vxx = torch.autograd.grad(torch.sum(Vx * r_f, dim=1).sum(), x, create_graph=True)[0]

        L_V = torch.sum(Vx * f, dim=1) + 0.5 * torch.sum((torch.sum(g * r_f, dim=1).view(-1, 1) * g) * r_Vxx, dim=1) # common noise
        # L_V = torch.sum(Vx * f, dim=1) + 0.5 * torch.sum(((g * r_f)[:,:dim]+(g * r_f)[:,dim:]) * ((g * r_Vxx)[:,:dim]+(g * r_Vxx)[:,dim:]), dim=1) # uncorrelated noise

        Vxg = torch.sum(Vx*g,dim=1) # common noise
        # Vxg = torch.norm((Vx*g)[:,:dim]+(Vx*g)[:,dim:],p=2,dim=1) # uncorrelated noise

        loss = -Vxg ** 2 / (V ** 2) + 2.5 * L_V / V
        loss = (F.relu(loss)).mean()


        # if loss == 0:
        #     break
        Loss.append(loss.item())
        if loss.item() == min(Loss):
            best_control_model = model._control.state_dict()
            best_V_model = model._lya.state_dict()
        print(out_iters,i, 'total loss=',loss.item(),"Lyapunov Risk=",loss.item(),model.W.sum()) #,model._control.controlnets[0].layer1.weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        i += 1
    '''
    save model 
    '''
    model._control.load_state_dict(best_control_model)

    # torch.save(torch.tensor(Loss), './data/loss_AIpara_{}_{}.pt'.format(mode_list[out_iters],model._control.upperbound))
    # torch.save(best_control_model,'./data/AIcontrol_{}_50_{}_2.5_{}.pkl'.format(mode_list[out_iters],N,model._control.upperbound))
    # torch.save(best_V_model, './data/V_{}_50_{}_2.5_{}.pkl'.format(mode_list[out_iters],N,model._control.upperbound))

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

    def controllerd(true_y0, dt, seed, model):
        np.random.seed(seed)
        N = 20000  # len(driving)
        sol,csol = torch.zeros([N, dim*2]),torch.zeros([N, dim*2])
        sol[0],csol[0] = true_y0,true_y0
        w = torch.from_numpy(np.random.normal(0, 1, [N,1])).repeat(1,2*dim) # common noise
        # w = torch.from_numpy(np.random.normal(0, 1, [N,2*dim])) # uncorrelated noise
        # w = torch.from_numpy(np.random.normal(0, 1, [N,dim])).repeat(1,2) # uncorrelated noise
        for i in range(N - 1):
            x = csol[i:i + 1]
            with torch.no_grad():
                u = model.FN_g2(0.0, x)[:, :, 0]
            # u = model.FN_linear_g(0.0, x)[:, :, 0]
            new_x = x + dt * model.FN(0.0,x) + w[i:i+1] * np.sqrt(dt) * u
            # new_x = x + dt * (model.FN(0.0, x)-model.strength*u)
            csol[i + 1:i + 2] = new_x
            y = sol[i:i + 1]
            new_y = y + dt * model.FN(0.0,y)
            sol[i + 1:i + 2] = new_y
        return sol,csol
    # true_y0 = true_y[-N:-N + 1, :]
    setup_seed(0)
    true_y0 = torch.randn([1, 2 * dim])  # 初值
    true_y,cont_y = controllerd(true_y0,1e-2,369,model)
    # true_y = originaL(true_y0,30000,1e-2,model)

    L = dim

    cont_y = cont_y.detach().numpy()
    true_y = true_y.detach().numpy()

    plt.subplot(131)
    T = 2000
    # for i in range(10):
    #     plt.plot(np.arange(T), cont_y[-T:, i],label='{}'.format(i))
    plt.imshow(cont_y[-10000:, 0:L].T, extent=[0, 500, 0, L], cmap='RdBu', aspect='auto')
    plt.title(r'Controlled Std={:.3f}'.format(np.std(cont_y[-2000:,:dim],axis=1).mean()))


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
    plt.title('Original Std={:.3f}'.format(np.std(true_y[-2000:,:dim],axis=1).mean()))




    out_iters+=1
end = timeit.default_timer()


print('total time: {:.5f}'.format(end-start))
plt.show()