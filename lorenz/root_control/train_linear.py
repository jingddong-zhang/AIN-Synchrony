import matplotlib.pyplot as plt
import torch
from functions import *


setup_seed(369)

N = 2000             # sample size
dim = 2
D_in = 3            # input dimension
H1 = 3*D_in             # hidden dimension
D_out = 3           # output dimension
x_ = torch.Tensor(N, 1).uniform_(-20, 20).requires_grad_(True)
y_ = torch.Tensor(N, 1).uniform_(-25, 25).requires_grad_(True)
z_ = torch.Tensor(N, 1).uniform_(0, 50).requires_grad_(True)
data = torch.cat((x_,y_,z_),dim=1)
data = torch.Tensor(N, D_in).uniform_(-5, 5).requires_grad_(True)
out_iters = 0


true_y0 = torch.randn([1, 3*dim])**2  # 初值

t = torch.linspace(0., 50., 2000)  # 时间点
model = Model_lorenz(D_in,H1,D_out,dim,0,(D_in,),[10,10,1])
with torch.no_grad():
    true_y = odeint(model.lorenz, true_y0, t, method='dopri5')[:,0,:]
# torch.save(true_y,'./data/train_data_50_2000.pt')

# true_y = torch.load('./data/train_data_50_2000.pt')
fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(true_y[:,0],true_y[:,2],true_y[:,4])
# print(max(true_y[:,0]),max(true_y[:,2]),max(true_y[:,4]),min(true_y[:,0]),min(true_y[:,2]),min(true_y[:,4]))
# for i in range(dim):
#     plt.plot(np.arange(len(true_y)),true_y[:,i])
# plt.show()

while out_iters < 5:
    # break
    start = timeit.default_timer()
    S = torch.cat((true_y[:,0:1],true_y[:,2:3],true_y[:,4:5]),dim=1)
    model = Model_lorenz(D_in,H1,D_out,dim,S,(D_in,),[2*H1,2*H1,1])
    i = 0
    # t = 0
    max_iters = 1000
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    Loss = []
    # theta = 0.75
    r_f = torch.randn(D_in)  # hutch estimator vector
    model_linear = Model_lorenz_linear(out_iters+1)
    while i < max_iters:
        # break
        x = data.requires_grad_(True)
        f = model.forward_nonauto(1., x)[:,0:3]
        # g = model.train_g(1.,x)[:,0:3,0]
        g = model_linear.train_linear_g(1.,x)[:,0:3,0]
        # x = x[:,0:3*dim]

        V = model._lya(x)
        Vx = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        r_Vxx = torch.autograd.grad(torch.sum(Vx * r_f, dim=1).sum(), x, create_graph=True)[0]
        L_V = torch.sum(Vx * f, dim=1) + 0.5 * torch.sum((torch.sum(g * r_f, dim=1).view(-1, 1) * g) * r_Vxx, dim=1)
        Vxg = torch.sum(Vx*g,dim=1)
        loss = -Vxg ** 2 / (V ** 2) + 2.5 * L_V / V
        loss = (F.relu(loss)).mean()



        # if loss == 0:
        #     break
        Loss.append(loss.item())
        if loss.item() == min(Loss):
            best_model = model._lya.state_dict()
        print(i, 'total loss=',loss.item(),"Lyapunov Risk=",loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        i += 1

    # torch.save(model._lya.state_dict(), './data/V_AI_2.5.pkl')
    # torch.save(torch.tensor(Loss),'./data/loss_AI.pt')

    # torch.save(model._lya.state_dict(), './data/V_linear_{}_100.pkl'.format(out_iters+1+10))
    torch.save(best_model, './data/V_linear_{}_100.pkl'.format(out_iters+1+10))
    torch.save(torch.tensor(Loss),'./data/loss_{}_100.pt'.format(out_iters+1+10))

    # torch.save(model._control.state_dict(),'./data/control_y_50_2000_100.pkl')
    # model._control.load_state_dict(torch.load('./data/control_y_50_2000_100.pkl'))


    # generate_AI(model,true_y0)
    # generate_linear(true_y0)
    # generate_linear_ode(true_y0)
    out_iters += 1

#     t = torch.linspace(0., 10., 5000)  # 时间点
#     DLC = Model_lorenz_linear(10)
#     torch.manual_seed(4) #10,11,13
#     # true_y0 = torch.Tensor(1,3*dim).uniform_(-1,1)
#     noise = torch.Tensor(1,3).uniform_(-2.5,2.5)
#     true_y0[:,0:6:2] += noise
#     with torch.no_grad():
#         # true_y = odeint(model.forward, test_y0, t, method='dopri5')[:,0,:]
#         # cont_y = odeint(DLC.f_control, true_y0, t, method='dopri5')[:,0,:]
#         cont_y = torchsde.sdeint(DLC, true_y0, t, method='euler', names={'drift': 'f_control', 'diffusion': 'non_g'})[:,
#                  0, :]
#         cont_y = torchsde.sdeint(DLC, cont_y[-2:-1,:], t, method='euler', names={'drift': 'f_control', 'diffusion': 'non_g'})[:,
#                  0, :]
#         # cont_y = torchsde.sdeint(model, true_y0, t, method='euler', names={'drift': 'lorenz', 'diffusion': 'lorenz_g'})[:,
#         #          0, :]
#         # cont_y = torchsde.sdeint(model, cont_y[-2:-1,:], t, method='euler', names={'drift': 'lorenz', 'diffusion': 'lorenz_g'})[:,
#         #          0, :]
#         # cont_y = torchsde.sdeint(model, cont_y[-2:-1,:], t, method='euler', names={'drift': 'lorenz', 'diffusion': 'lorenz_g'})[:,
#         #          0, :]
#     state = cont_y
#     L = dim
#     st = torch.zeros([len(state), 3])
#     x,y,z = state[:,0:L],state[:,L:2*L],state[:,2*L:3*L]
#     st[:, 0] = x[:,1]-x[:,0]
#     st[:, 1] = y[:,1]-y[:,0]
#     st[:, 2] = z[:,1]-z[:,0]
#     u = model._control(st).detach().numpy()
#     print(cont_y[:,0],'\n',cont_y[:,1])
#     plt.subplot(131)
#     for i in range(dim):
#         plt.plot(np.arange(len(cont_y)), cont_y[:, i ])
#     plt.title(r'$x_{1,2}$')
#     plt.subplot(132)
#     plt.plot(np.arange(len(cont_y)), cont_y[:, 1]-cont_y[:,0])
#     plt.title(r'$x_2-x_1$')
#     plt.subplot(133)
#     plt.plot(np.arange(len(cont_y)), u[:, 1])
#     plt.title(r'$u$')
#     # plt.ylim(-10,10)
#
#
#
#     out_iters+=1
# #
# plt.show()