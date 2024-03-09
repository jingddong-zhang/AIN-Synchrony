import matplotlib.pyplot as plt

from functions import *

def generate(model,true_y0,g_case='lorenz_g'):
    N = 5000
    m = 10
    t = torch.linspace(0., 10., N)  # 时间点
    data = torch.zeros([m, 2 * N, 6])
    for i in range(m):
        torch.manual_seed(i+1)
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
    torch.save(data, './data/data_root_AI.pt')


# dim = 2
# D_in = 3            # input dimension
# H1 = 3*D_in             # hidden dimension
# D_out = 3           # output dimension
# model = Model_lorenz(D_in,H1,D_out,dim,0,(D_in,),[2*H1,2*H1,1])
# model._control.load_state_dict(torch.load('./data/control_y_50_2000.pkl'))
# generate(model)

def check():
    data = torch.load('./data/data_leaf_AI.pt')

    seed = 6
    plt.plot(np.arange(len(data[0])), data[seed,:, 1] - data[seed,:, 0])
    plt.title(r'$x_2-x_1$')
    plt.show()