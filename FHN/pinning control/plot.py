import timeit
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np
import torch
from functions import *
import networkx as nx
from FVS import find_FVS

import networkx as nx
torch.set_default_dtype(torch.float64)

colors = [
    [98 / 255, 190 / 255, 166 / 255], #青色
    [107/256,	161/256,255/256], # #6ba1ff
    [255/255, 165/255, 0],
    [233/256,	110/256, 248/256], # #e96eec
    # [0.6, 0.6, 0.2],  # olive
    # [0.5333333333333333, 0.13333333333333333, 0.3333333333333333],  # wine
    # [0.8666666666666667, 0.8, 0.4666666666666667],  # sand
    # [223/256,	73/256,	54/256], # #df4936

    [0.55, 0.71, 0.0], # applegreen
    [0.0, 0.0, 1.0],  # ao
    # [0.4, 1.0, 0.0], # brightgreen
    [0.99, 0.76, 0.8], # bubblegum
    [0.93, 0.53, 0.18], # cadmiumorange
    [11/255, 132/255, 147/255], # deblue
    [204/255, 119/255, 34/255], # {ocra}
    [0.6, 0.4, 0.8], # amethyst
    [31/255,145/255,158/255],
    [127/255,172/255,204/255],
    [233/255,108/255,102/255],
]

seven_color = [[75 / 255, 102 / 255, 173 / 255], [98 / 255, 190 / 255, 166 / 255],
               [205 / 255, 234 / 255, 157 / 255], 'gold',
               [253 / 255, 186 / 255, 107 / 255], [235 / 255, 96 / 255, 70 / 255], [163 / 255, 6 / 255, 67 / 255]
               ]

def pnas_order(data, T=10000,dt=1000, delta=0.1):
    L = int(data.shape[1] / 2)
    data = data[T:T+dt, 0:L]
    mean1 = np.mean(data, axis=1)
    std1 = np.std(mean1)
    std2 = np.mean(np.std(data, axis=0))
    order = std1 / std2
    return order


def plot():

    fontsize = 25
    ticksize = 25

    rc_fonts = {
        "text.usetex": True,
        'text.latex.preview': True,  # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}""", r"""\usepackage{amsmath}""", r"""\usepackage{amsfonts}"""],
        'font.sans-serif': 'Times New Roman'
    }
    import matplotlib
    matplotlib.rcParams.update(rc_fonts)

    A = np.load('./data/topology/39 Di_ER k=4 seed=369.npy')

    dim = len(A)
    indegree = A.sum(axis=1)
    mode1 = np.where(indegree == 0)[0]
    FVS = np.array(find_FVS(A))
    mode2 = np.concatenate((mode1, FVS)).tolist()


    fig = plt.figure(figsize=(15, 6))
    plt.subplots_adjust(left=0.03, bottom=0.1, right=0.95, top=0.95, hspace=0.1, wspace=0.2)



    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    import matplotlib.cm

    ax = plt.subplot(1,3,1)
    G = nx.from_numpy_array(A)
    node_colors = list(np.arange(dim))
    for i in range(dim):
        if i in mode2:
            node_colors[i] = colors[1]
        else:
            node_colors[i] = colors[2]


    nx.set_node_attributes(G, dict(zip(G.nodes(), node_colors)), 'color')

    edgeWeig = [G.edges[i]['weight'] for i in G.edges()]

    options = {
        'node_size': 100,
        "node_color": node_colors,  # "#A0CBE2",
        "width": 1.5,
        # 'alpha': 0.5,
        "edge_cmap":   mpl.cm.get_cmap('jet'),
        'edge_color': edgeWeig,
        'edge_vmin': -0.4,
        'edge_vmax': 0.4,
        "with_labels": False,
    }
    nx.draw(G, pos=nx.kamada_kawai_layout(G), **options)
    plt.text(-1.0, 0.6, '(a)', fontsize=fontsize)

    plt.subplot(132)
    loss = torch.load('./data/loss_u_max=20.0 b=2.1.pt').detach().numpy()
    plt.plot(np.arange(len(loss)),loss,c='b')
    plt.xlabel('Epoch',fontsize=fontsize,labelpad=-25)
    plt.ylabel('Loss',fontsize=fontsize,labelpad=-15)
    plt.xticks([0,500],fontsize=fontsize)
    plt.yticks([0,1],fontsize=fontsize)
    plt.text(0,1.0,'(b)',fontsize=fontsize)

    plt.subplot(133)
    cont_data = np.load('./data/cont u_max=20.0 time=10000.npy')[:800000]
    true_data = np.load('./data/true.npy')[:800000]
    L = cont_data.shape[0]
    order_parameter = []
    true_order_parameter = []
    dt = 5000
    for k in range(int(L/dt)):
        order_parameter.append(pnas_order(cont_data,int(k*dt),dt))
        true_order_parameter.append(pnas_order(true_data, int(k * dt),dt))
    order_parameter = np.array(order_parameter)
    true_order_parameter = np.array(true_order_parameter)

    plt.plot(np.linspace(50, 8000-50, len(order_parameter)), order_parameter, color=colors[1],label='Controlled')
    # plt.fill_between(np.linspace(0.5, 120-0.5, len(mean)), mean - std ** 2, mean + std ** 2, color=colors[1], alpha=0.4)
    plt.plot(np.linspace(50, 8000-50, len(true_order_parameter)), true_order_parameter, color=colors[2],label='Original')
    # plt.fill_between(np.linspace(0.5, 120-0.5, len(mean)), true_mean - true_std ** 2, true_mean + true_std ** 2, color=colors[2], alpha=0.4)
    plt.xticks([0,8000],fontsize=ticksize)
    plt.yticks([0.7, 1], fontsize=ticksize)
    plt.xlabel('Time',fontsize=fontsize,labelpad=-25)
    plt.ylabel(r'$R$',fontsize=fontsize,labelpad=-25)
    plt.legend(loc=1,frameon=False,fontsize=fontsize)
    plt.text(0., 0.97, '(c)', fontsize=fontsize)








plot()

plt.show()