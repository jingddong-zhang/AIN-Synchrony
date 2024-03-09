import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import networkx as nx
import pandas as pd
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from functions import *

colors = [
    [107/256,	161/256,255/256], # #6ba1ff
    [255/255, 165/255, 0],
    [233/256,	110/256, 248/256], # #e96eec
    # [0.6, 0.6, 0.2],  # olive
    # [0.5333333333333333, 0.13333333333333333, 0.3333333333333333],  # wine
    # [0.8666666666666667, 0.8, 0.4666666666666667],  # sand
    # [223/256,	73/256,	54/256], # #df4936
    [0.6, 0.4, 0.8], # amethyst
    [0.0, 0.0, 1.0], # ao
    [0.55, 0.71, 0.0], # applegreen
    # [0.4, 1.0, 0.0], # brightgreen
    [0.99, 0.76, 0.8], # bubblegum
    [0.93, 0.53, 0.18], # cadmiumorange
    [11/255, 132/255, 147/255], # deblue
    [204/255, 119/255, 34/255], # {ocra}
    [31/255,145/255,158/255],
    [127/255,172/255,204/255],
    [233/255,108/255,102/255],
]
colors = np.array(colors)
def plot_grid():
    plt.grid(b=True, which='major', color='gray', alpha=0.6, linestyle='dashdot', lw=1.)
    # minor grid lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='beige', alpha=0.8, ls='-', lw=1)
    # plt.grid(b=True, which='both', color='beige', alpha=0.1, ls='-', lw=1)
    pass


def normalize(data):
    return data / data.max()

def process_data(data):
    '''
    :param data: size(m,N.dim*2+2)
    :return: trajectories without nan
    '''
    a = data[:,-1,0]
    b = torch.isnan(a)
    seed_list = []
    for _ in enumerate(b):
        if _[1] == False:
            seed_list.append(_[0])
    res = data[seed_list]
    return res

def plot_v1():
    import matplotlib
    from matplotlib.patches import ConnectionPatch
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    rc_fonts = {
        "text.usetex": True,
        'text.latex.preview': True,  # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}""", r"""\usepackage{amsmath}""", r"""\usepackage{amsfonts}"""],
        'font.sans-serif': 'Times New Roman'
    }
    matplotlib.rcParams.update(rc_fonts)
    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams['ytick.direction']='in'
    plt.rcParams['xtick.direction'] = 'in'
    fontsize = 18
    ticksize = 15
    textsize = 20
    xlabelpad = -10

    # LC = torch.load('./data/data_root_linear_ode_100.pt') # k,m,N,dim
    LC = torch.load('./data/data_root_linear_100_15_50_v0.pt')  # k,m,N,dim
    AIC1 = torch.load('./data/data_root_AI_100_50.pt') # m,N,dim
    AIC2 = torch.load('./data/data_leaf_AI_100_50.pt').numpy() # m,N,dim
    AICx = torch.load('./data/data_AIx_100_50.pt').numpy()  # m,N,dim
    seed_ = [0,1,2,3,4,5,6,7,8,9]
    LC = LC[:,:,:,:].numpy()
    AIC1 = AIC1[:,:,:].numpy()
    print(LC.shape,type(LC))
    K,m,N = LC.shape[0],LC.shape[1],LC.shape[2]
    T = 20
    dt = T/N


    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.03, bottom=0.11, right=0.95, top=0.98, hspace=0.25, wspace=0.5) #right=0.95,wspace=0.7
    gs = GridSpec(3, 7, figure=fig)

    fig.add_subplot(gs[0,0:2]) #########################################################################################
    plt.grid(None)
    plt.axis('off')
    plt.text(0.5, 0.9, '(a)', fontsize=textsize)


    fig.add_subplot(gs[1,0:3]) #########################################################################################
    succ_rate = np.load('./data/succ_rate.npy')
    seven_color = [[75 / 255, 102 / 255, 173 / 255], [98 / 255, 190 / 255, 166 / 255],
                   [205 / 255, 234 / 255, 157 / 255], 'gold',
                   [253 / 255, 186 / 255, 107 / 255], [235 / 255, 96 / 255, 70 / 255], [163 / 255, 6 / 255, 67 / 255]
                   ]
    index = [r'$\Omega_{}$'.format(i + 1) for i in range(7)] + [r'AIN$y$']
    plt.bar(index, np.append(np.mean(succ_rate, axis=1), 1.0), width=0.6, color=seven_color, alpha=0.6, lw=2.0,
            edgecolor=seven_color)
    for i in range(7):
        plt.errorbar(index[i:i + 1], np.mean(succ_rate, axis=1)[i], np.std(succ_rate, axis=1)[i], ecolor=seven_color[i],
                     capsize=6)
        plt.scatter(np.random.uniform(-0.2 + 1.0 * i, 0.2 + 1.0 * i, 6), succ_rate[i], color=seven_color[i])
    plt.yticks([0, 1], ['0', '1'], fontsize=ticksize)
    # plt.yticks([0, 1], ['0', r'100$\%$'], fontsize=ticksize)
    plt.ylabel('Succ. Rate', fontsize=fontsize, labelpad=-10)
    plt.xticks(fontsize=ticksize)
    plt.text(-0.5,0.85,'(b)',fontsize=textsize)

    fig.add_subplot(gs[2,0:3]) #########################################################################################
    def state(data, T):
        diff = np.sqrt(np.sum((data[:, :, 0:6:2] - data[:, :, 1:6:2]) ** 2, axis=2))
        mean = np.mean(diff, axis=0)
        return mean[T]

    state_list = []
    end_time = 5999
    for j in range(K):
        x = LC[j, :]
        state_list.append(state(x, end_time))
    state_list.append(state(AIC1, end_time))
    # state_list = normalize(torch.tensor(state_list))
    state_list = normalize(np.array(state_list))


    control = ControlNet(3,3*3,3,2)
    control.load_state_dict(torch.load('./data/control_y_50_2000.pkl'))
    def energy(data):
        from scipy import integrate
        t = np.linspace(0, T, N)
        e = np.zeros(m)
        for i in range(m):
            e[i] = integrate.trapz(data[i,:]**2, t)
        return e
    energy_list = []
    model = Model_lorenz_linear(0)
    for j in range(K):
        k = j+1
        x = LC[j,:]
        LC_con = k*np.sum((x[:,:,1:6:2]-x[:,:,0:6:2]),axis=2)
        LC_con = model.bound(torch.from_numpy(LC_con)).numpy()
        energy_list.append(energy(LC_con).mean())
    rootC = np.zeros([m,N])
    for i in range(m):
        rootC[i] = control(torch.from_numpy(AIC1[i,:,1:6:2]-AIC1[i,:,0:6:2]))[:,1].detach().numpy()
    rootE = energy(rootC).mean()
    energy_list.append(rootE)
    energy_ = normalize(torch.tensor(energy_list))

    def std(data, T):
        diff = np.sqrt(np.sum((data[:, T:, 0:6:2] - data[:, T:, 1:6:2]) ** 2, axis=2))
        std0 = np.std(diff, axis=0)
        std1 = np.mean(std0)
        return std1

    data = torch.load('./data/eigvalstrace_20_v1.pt')
    mean = normalize(torch.mean(data,dim=1))
    # index_list = [6,7,8,9,10,11,12,13,14]
    index_list = [i for i in range(10,16)]
    convex = torch.zeros(len(index_list)+1)

    # plt.scatter(np.arange(5),mean[:5])
    # plt.axhline(mean[-1], ls='--', color=colors[-3], label='AC Root')
    loss = torch.zeros(len(index_list)+1)
    std_list = torch.zeros(len(index_list)+1)
    for i in range(len(index_list)):
        loss[i] = torch.load('./data/loss_{}_100_b.pt'.format(index_list[i])).min()
        convex[i] = mean[index_list[i]-1]
        std_list[i] = std(LC[index_list[i]-1],3000)
    loss[-1] = torch.load('./data/loss_AI.pt').min()
    convex[-1] = mean[-1]
    std_list[-1] = std(AIC1,3000)
    # loss = normalize(loss)
    # plt.scatter(np.arange(5), loss[:5])
    co = [[30/255,21/255,72/255],[255/255,46/255,76/255],[46/255,153/255,176/255],[252/255,215/255,127/255]]
    types = ['LC{}'.format(_) for _ in index_list]+[r'AIN$y$']
    bar_width = 0.3
    index_male = np.arange(len(index_list)+1)
    index_female = index_male + bar_width

    plt.bar(index_male, height=normalize(1/std_list), width=bar_width, color=seven_color[4], alpha=0.6,label='Stability',lw=2.0,edgecolor=seven_color[4]) #ECA47C
    plt.bar(index_male+bar_width, height=convex, width=bar_width, color=seven_color[5],alpha=0.6, label='Convexity',lw=2.0,edgecolor=seven_color[5]) #759DDB
    plt.legend(loc=1,fontsize=ticksize,ncol=1,frameon=False,bbox_to_anchor=[0.85,1.0])  # 显示图例
    plt.xticks(index_male + bar_width / 2, types,fontsize=ticksize,rotation=45)
    plt.yticks([0, 1], ['0', '1'], fontsize=ticksize)
    plt.text(-0.3,0.85,'(e)',fontsize=textsize)

    ax_0 = fig.add_subplot(gs[1, 5:7]) #########################################################################################
    plt.plot(np.arange(K), state_list[:K], c=colors[0],marker='o') # co[2]
    plt.axhline(state_list[-1], ls='--', color=colors[0], label=r'AI$y$')
    plt.xticks([0, 14], ['1', '15'],fontsize=ticksize)
    plt.yticks([0, 1], ['0', '1'],fontsize=ticksize)
    plt.xlabel(r'$k$',labelpad=xlabelpad,fontsize=fontsize)
    plt.ylabel(r'$\Vert\xi(20)\Vert$',labelpad=xlabelpad,fontsize=fontsize,c=colors[0])
    plt.text(11.5, 0.85, '(d)',fontsize=textsize)

    ax_1 = ax_0.twinx()
    plt.plot(np.arange(K),energy_[:K],c=colors[1],marker='o') # co[3]
    plt.axhline(energy_[-1], ls='--', color=colors[1], label=r'AI$y$')
    plt.xticks([0, 14], ['1', '15'],fontsize=ticksize)
    plt.yticks([0, 1], ['0', '1'],fontsize=ticksize)
    plt.xlabel(r'$k$',labelpad=xlabelpad,fontsize=fontsize)
    plt.ylabel('Energy',labelpad=xlabelpad+5,fontsize=fontsize,c=colors[1])
    # plt.legend(fontsize=ticksize)


    fig.add_subplot(gs[1,3:5]) #########################################################################################
    def plot1(data,color,label,end_time=end_time):
        end_time = end_time+1
        diff = np.sqrt(np.sum((data[:,:end_time,0:6:2] - data[:,:end_time,1:6:2]) ** 2,axis=2))
        mean,std = np.mean(diff,axis=0),np.std(diff,axis=0)
        plt.fill_between(np.arange(end_time) * dt, mean - std, mean + std, color=color, alpha=0.4)
        plt.plot(np.arange(end_time) * dt, mean, color=color,label=label)
        plt.xticks([0,12],fontsize=ticksize)
        plt.yticks([0, 300],fontsize=ticksize)

    plot1(LC[-1,:],colors[-1],'LC10')
    plot1(AIC1, colors[-2],r'AIN$y$')
    plot1(AICx, colors[2], r'AIN$x$')
    plot1(AIC2, colors[1], r'AIN$xz$')
    # plt.axhline(5, ls='--',color=colors[-3],label='Safe Line')
    plt.legend(loc=5,fontsize=ticksize, frameon=False,handlelength=1.0,ncol=2,columnspacing=0.5)
    plt.xlabel('Time', fontsize=fontsize,labelpad=xlabelpad)
    plt.ylabel('MSE', fontsize=fontsize,labelpad=xlabelpad-15)
    plt.text(0.2, 275.0, '(c)',fontsize=textsize)
    plot_grid()


    ax = fig.add_subplot(gs[0,3:7],projection='3d') #########################################################################################
    ax.grid(None)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.axis('off')
    data_3d = AIC1[0,-5000:]
    ax.plot3D(data_3d[:,0],data_3d[:,2],data_3d[:,4],color=colors[2],label='Driving',lw=2.0)
    ax.plot3D(data_3d[:,1],data_3d[:,3],data_3d[:,5],ls=(0,(5,3)), color='black',label='Response',lw=1.5)
    # plt.legend(fontsize=ticksize,frameon=False,bbox_to_anchor=(0.8,0.3),ncol=1)


    # chartBox = ax.get_position()
    # ax.set_position([chartBox.x0,
    #                   chartBox.y0-0.1,
    #                   chartBox.width*1.3,
    #                   chartBox.height*1.5])



    ax_2 = fig.add_subplot(gs[2,5:7]) #########################################################################################
    V_eigvals = torch.load('./data/eigvalsmean_AI_root_training_2000.pt').mean(axis=1)
    control_energy = np.load('./data/energy_AI_root_traing_2000.npy')
    from audtorch.metrics.functional import pearsonr
    corr = pearsonr(torch.from_numpy(control_energy).view(1,-1), V_eigvals.view(1,-1))
    plt.plot(np.arange(len(V_eigvals)),V_eigvals,c=colors[0],marker='o')
    plt.ylabel('Convexity',fontsize=fontsize,labelpad=-15,c=colors[0])
    plt.yticks([4,12],fontsize=ticksize)
    plt.xticks([0,10,20],['0','1','2'],fontsize=ticksize)
    plt.xlabel(r'Epochs $[\times 10^2]$',fontsize=fontsize,labelpad=-5)
    plt.text(10,8.5,r'$\rho={:.2f}$'.format(corr[0,0].item()),fontdict=dict(fontsize=fontsize, color='black',),
             bbox={'facecolor': '#F7EAD5', #填充色
          'edgecolor':'black',#外框色
           'alpha': 1.0, #框透明度
           'pad': 4,#本文与框周围距离
          })
    plt.text(2.7, 11.2, '(g)', fontsize=textsize)
    ax_3 = ax_2.twinx()
    plt.plot(np.arange(len(control_energy)),control_energy,c=colors[1],marker='o')
    plt.yticks([10000,90000],['1','9'],fontsize=ticksize)
    plt.ylabel(r'Energy $[\times 10^4]$',fontsize=fontsize,c=colors[1],labelpad=xlabelpad+5)
    # plt.title(r'Pearson: ${:.2f}$'.format(corr[0,0].item()))

    ax = fig.add_subplot(gs[2,3:5]) #########################################################################################
    V_model1 = ICNN((3,), [18, 18, 1], 0.1, 1e-3)  # D_in = 3 , H1 = 3*D_in
    V_model2 = ICNN((3,), [18, 18, 1], 0.1, 1e-3)  # D_in = 3 , H1 = 3*D_in

    vnorm = mpl.colors.Normalize(vmin=0, vmax=50)

    def generate_dir(seed):
        torch.manual_seed(seed)
        e1 = torch.randn(3)
        e1 = e1 / torch.linalg.norm(e1)
        e2 = torch.zeros_like(e1)
        e2[0], e2[1], e2[2] = -e1[2], 0., e1[0]
        return torch.cat((e1.view(1, 3), e2.view(1, 3)), dim=0)

    def draw_image(V1, V2, direction_base):
        '''
        :param V_model d\to1:
        :param direction_base:(2,d),2个随机选取的正交方向
        :return: V
        '''
        n = 500
        l = 3
        with torch.no_grad():
            x = torch.linspace(-l, l, n)
            y = torch.linspace(-l, l, n)
            X, Y = torch.meshgrid(x, y)
            inp = torch.stack([X, Y], dim=2)  # 欧式空间在标准坐标方向的规则采样
            inp = torch.mm(inp.view(-1, 2), direction_base)  # 转化到d维空间的样本
            image = V1(inp) - V2(inp)
            image = image[..., 0].view(n, n).detach().cpu()
        h = plt.contourf(X, Y, image, 25, cmap=mpl.cm.jet,norm=vnorm)
        # plt.imshow(image, extent=[-l, l, -l, l], aspect='auto',cmap='rainbow',norm=vnorm)
        # plt.xlabel(r'$\theta$')
        plt.xticks([-2, 2],fontsize=ticksize)
        plt.yticks([-2,2],fontsize=ticksize)
        plt.xlabel(r'$\vec{e}_1$',labelpad=xlabelpad-5,fontsize=fontsize)
        plt.ylabel(r'$\vec{e}_2$', labelpad=xlabelpad-10,fontsize=fontsize)
        plt.text(-2.7, 1.85, '(f)', fontsize=textsize)
        # cax = fig.add_axes([ax.get_position().x1 + 0.02, ax.get_position().y0, 0.01, ax.get_position().height])
        cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-0.065,ax.get_position().width, 0.02])
        cb = plt.colorbar(h, cax=cax,orientation='horizontal')
        # cb.set_label(r'$v_i$', fontdict={'size':fontsize})
        # cb.ax.set_title(r'$v_i$', fontsize=fontsize)
        cb.set_ticks([40, 0])
        cb.ax.tick_params(labelsize=ticksize)

        # plt.clim(-2, 2)
        return image

    V_model1.load_state_dict(torch.load('./data/V_AI_2.5.pkl'))
    V_model2.load_state_dict(torch.load('./data/V_linear_10_100.pkl'))
    draw_image(V_model1, V_model2, generate_dir(9))




plot_v1()

def plot_sub():
    AIC1 = torch.load('./data/data_root_AI_100_50.pt') # m,N,dim
    AIC1 = AIC1[:,:,:].numpy()


    fig = plt.figure(figsize=(5, 5))
    # plt.subplots_adjust(left=0.03, bottom=0.11, right=0.95, top=0.98, hspace=0.25, wspace=0.5) #right=0.95,wspace=0.7
    # gs = GridSpec(3, 7, figure=fig)
    ax = fig.add_subplot(111,
                         projection='3d')  #########################################################################################
    ax.grid(None)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.axis('off')
    data_3d = AIC1[0, -5000:]
    ax.plot3D(data_3d[:, 0], data_3d[:, 2], data_3d[:, 4], color=colors[2], label='Driving', lw=2.0)
    ax.plot3D(data_3d[:, 1], data_3d[:, 3], data_3d[:, 5], ls=(0, (5, 3)), color='black', label='Response', lw=1.5)
    # plt.legend(fontsize=ticksize,frameon=False,bbox_to_anchor=(0.8,0.3),ncol=1)

# plot_sub()
plt.show()