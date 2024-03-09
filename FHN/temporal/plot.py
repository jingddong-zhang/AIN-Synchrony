import timeit
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np
import torch
from functions import *
import networkx as nx

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

def pnas_order(data, T=10000, delta=0.1):
    L = int(data.shape[1] / 2)
    data = data[T:T+100, 0:L]
    mean1 = np.mean(data, axis=1)
    std1 = np.std(mean1)
    std2 = np.mean(np.std(data, axis=0))
    order = std1 / std2
    return order

def temp_A(dim,t):
    k = 1.0
    w = 1.0
    A = np.zeros([dim,dim])
    for i in range(dim):
        for j in range(dim):
            if j != i:
                if i<=int((dim+1)/2)-1 and j<=int((dim+1)/2)-1:
                    A[i, j] = (1.0 + (6.0 - 8.0 / (dim + 1)) * k * np.sin(w * t))/dim
                else:
                    A[i, j] = (1-2*k*np.sin(w * t))/dim
    return A

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

    fig = plt.figure(figsize=(15, 9))
    plt.subplots_adjust(left=0.03, bottom=0.07, right=0.98, top=0.95, hspace=0.1, wspace=0.2)
    dim = 11


    t_list = [math.pi/2,math.pi,math.pi*3/2]
    t_label_list = [r'$t=\frac{\pi}{2}$',r'$t=\pi$',r'$t=\frac{3\pi}{2}$']
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    import matplotlib.cm

    for i in range(3):
        ax = plt.subplot(2,3,i+1)
        A = temp_A(dim,t_list[i])
        G = nx.from_numpy_array(A)
        edgeWeig = [G.edges[i]['weight'] for i in G.edges()]
        print(edgeWeig)
        options = {
            'node_size': 100,
            "node_color": colors[2],  # "#A0CBE2",
            "width": 1.5,
            # 'alpha': 0.5,
            "edge_cmap":   mpl.cm.get_cmap('jet'),
            'edge_color': edgeWeig,
            'edge_vmin': -0.4,
            'edge_vmax': 0.4,
            "with_labels": False,
        }
        nx.draw(G, pos=nx.circular_layout(G), **options)
        if i == 0:
            plt.text(-1.1, 1.0, '(a)', fontsize=fontsize)
        plt.title(t_label_list[i],fontsize=fontsize)
    # plt.text(0.0, 1.0, 'Time', fontsize=fontsize)

    data = np.load('./data/u_max=5.0 m=10.npy',allow_pickle=True).item()
    cont_data = data['cont']
    true_data = data['true']
    m = cont_data.shape[0]
    L = cont_data.shape[1]
    order_parameter = []
    true_order_parameter = []
    for i in range(m):
        order_list = []
        true_order_list = []
        for k in range(int(L/100)):
            order_list.append(pnas_order(cont_data[i],int(k*100)))
            true_order_list.append(pnas_order(true_data[i], int(k * 100)))
        order_parameter.append(order_list)
        true_order_parameter.append(true_order_list)
    order_parameter = np.array(order_parameter)
    true_order_parameter = np.array(true_order_parameter)
    mean = order_parameter.mean(axis=0)
    std = order_parameter.std(axis=0)
    true_mean = true_order_parameter.mean(axis=0)
    true_std = true_order_parameter.std(axis=0)
    plt.subplot(234)
    plt.plot(np.linspace(0.5, 120-0.5, len(mean)), mean, color=colors[1],label='Controlled')
    plt.fill_between(np.linspace(0.5, 120-0.5, len(mean)), mean - std ** 2, mean + std ** 2, color=colors[1], alpha=0.4)
    plt.plot(np.linspace(0.5, 120-0.5, len(mean)), true_mean, color=colors[2],label='Original')
    plt.fill_between(np.linspace(0.5, 120-0.5, len(mean)), true_mean - true_std ** 2, true_mean + true_std ** 2, color=colors[2], alpha=0.4)
    plt.xticks([0,120],fontsize=ticksize)
    plt.yticks([0, 1], fontsize=ticksize)
    plt.xlabel('Time',fontsize=fontsize,labelpad=-15)
    plt.ylabel(r'$R$',fontsize=fontsize,labelpad=-15)
    plt.legend(loc=3,frameon=False,fontsize=fontsize)
    plt.text(0.0, 0.93, '(b)', fontsize=fontsize)

    cont_y = data['cont'][0]
    true_y = data['true'][0]

    plt.subplot(235)
    plt.imshow(true_y[:, 0:dim].T, extent=[0, 120, 0, dim], cmap='RdBu', aspect='auto')
    plt.xticks([0,120],fontsize=ticksize)
    plt.yticks([0,11],fontsize=ticksize)
    plt.xlabel('Time',fontsize=fontsize,labelpad=-15)
    plt.ylabel(r'$v_i$',fontsize=fontsize,labelpad=-25)
    plt.title('Original',fontsize=fontsize)
    plt.text(1, 10, '(c)', fontsize=fontsize)

    plt.subplot(236)
    plt.imshow(cont_y[:, 0:dim].T, extent=[0, 120, 0, dim], cmap='RdBu', aspect='auto')
    plt.xticks([0,120],fontsize=ticksize)
    plt.yticks([0,11],fontsize=ticksize)
    plt.xlabel('Time',fontsize=fontsize,labelpad=-15)
    plt.title('Controlled',fontsize=fontsize)
    plt.text(1, 10, '(d)', fontsize=fontsize)




plot()

plt.show()