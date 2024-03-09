import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import os.path as osp
import os
import torch
import networkx as nx
import re
from scipy.interpolate import CubicSpline
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

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

def pnas_order(data, T=15000, delta=0.1):
    L = int(data.shape[1] / 2)
    data = data[T:, 0:L]
    mean1 = np.mean(data, axis=1)
    std1 = np.std(mean1)
    std2 = np.mean(np.std(data, axis=0))
    order = std1 / std2
    return order

def plot():
    files = os.listdir('./data/results/128')

    fontsize = 18
    ticksize = 18
    textsize = 20
    seven_color = [[75 / 255, 102 / 255, 173 / 255], [98 / 255, 190 / 255, 166 / 255],
                   [205 / 255, 234 / 255, 157 / 255], 'gold',
                   [253 / 255, 186 / 255, 107 / 255], [235 / 255, 96 / 255, 70 / 255], [163 / 255, 6 / 255, 67 / 255]
                   ]
    rc_fonts = {
        "text.usetex": True,
        'text.latex.preview': True,  # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}""", r"""\usepackage{amsmath}""", r"""\usepackage{amsfonts}"""],
        'font.sans-serif': 'Times New Roman'
    }
    import matplotlib
    matplotlib.rcParams.update(rc_fonts)

    fig = plt.figure(figsize=(15, 6))
    plt.subplots_adjust(left=0.03, bottom=0.07, right=0.98, top=0.97, hspace=0.95, wspace=0.4)
    gs = GridSpec(8, 8, figure=fig)

    fig.add_subplot(gs[0:8,0:3]) ####################################################################################################################
    def analyze_0():
        no_noise = []
        common_noise = []
        uncorr_noise = []
        subfiles = [_ for _ in files if 'u_max' in _]
        for filename in subfiles:
            sub_ord_cont,sub_ord_true = [],[]
            data = np.load(osp.join('./data/results/128/',filename),allow_pickle=True).item()
            cont,true = data['cont'],data['true']
            for k in range(len(cont)):
                sub_ord_cont.append(pnas_order(cont[k]))
                sub_ord_true.append(pnas_order(true[k]))
            if 'mode=0' in filename:
                common_noise.append(sub_ord_cont)
            if 'mode=1' in filename:
                uncorr_noise.append(sub_ord_cont)
            no_noise.append(sub_ord_true)

        no_noise = np.array(no_noise)
        common_noise = np.array(common_noise)
        uncorr_noise = np.array(uncorr_noise)
        no_noise = np.array([no_noise.mean(axis=1),no_noise.std(axis=1)])[:,:3]
        common_noise = np.array([common_noise.mean(axis=1),common_noise.std(axis=1)])
        uncorr_noise = np.array([uncorr_noise.mean(axis=1),uncorr_noise.std(axis=1)])
        np.save('./data/results/128/analyze_u',{'no':no_noise,'comm':common_noise,'unc':uncorr_noise})
    # analyze_0()
    data = np.load('./data/results/128/analyze_u.npy',allow_pickle=True).item()
    no_noise,common_noise,uncorr_noise = data['no'],data['comm'],data['unc']
    index = ['Low','Middle','High']
    bar_width = 0.2

    plt.bar(np.arange(len(index))+bar_width*0,no_noise[0] , width=bar_width, color=seven_color[0], alpha=0.6, lw=2.0,label='No Noise')
    plt.bar(np.arange(len(index))+bar_width*1,common_noise[0] , width=bar_width, color=seven_color[1], alpha=0.6, lw=2.0,label='Common')
    plt.bar(np.arange(len(index))+bar_width*2,uncorr_noise[0] , width=bar_width, color=seven_color[6], alpha=0.6, lw=2.0,label='Uncorrelated')
    for i in range(3):
        plt.errorbar((np.arange(len(index))+bar_width*0)[i:i+1],no_noise[0,i], no_noise[1,i], ecolor=seven_color[0],capsize=5)
        plt.errorbar((np.arange(len(index))+bar_width*1)[i:i+1],common_noise[0,i], common_noise[1,i], ecolor=seven_color[1],capsize=5)
        plt.errorbar((np.arange(len(index))+bar_width*2)[i:i+1],uncorr_noise[0,i], uncorr_noise[1,i], ecolor=seven_color[6],capsize=5)

    plt.xticks(np.arange(len(index))+bar_width,index,fontsize=ticksize)
    plt.yticks([0,1],['0','1'],fontsize=ticksize)
    plt.ylim(0,1.2)
    plt.legend(ncol=1,fontsize=ticksize,frameon=False)
    plt.ylabel(r'$R_{150:200}$',fontsize=fontsize,labelpad=-13)
    plt.text(2.3,1.09,'(a)',fontsize=textsize)

    ax0 = fig.add_subplot(gs[0:4,3:6]) ####################################################################################################################
    xyA = (0.85,0.85)
    xyB = (0.85,0.65)
    coordsA = ax0.transData
    coordsB = ax0.transData
    con0 = ConnectionPatch(xyA, xyB,
                           coordsA, coordsB,
                           arrowstyle="->",
                           shrinkA=5, shrinkB=5,
                           mutation_scale=40,
                           fc=[199 / 255, 105 / 255, 142 / 255], color='red', alpha=0.5) #'salmon'
    con0.set_linewidth(5)
    ax0.add_artist(con0)
    subfiles1 = [_ for _ in files if 'G_max' in _]
    key_list = ['bay','ER','SF1','SF2'][::-1]
    label_list = ['Baydry','ER','SF1','SF2'][::-1]
    u_max_list = ['{:.1f}'.format(i*0.1+0.1) for i in range(30)]

    for i in range(4):
        key = key_list[i]
        subfiles2 = [_ for _ in subfiles1 if key in _]
        def analyze_1():
            mean,std = np.zeros(len(u_max_list)),np.zeros(len(u_max_list))
            for k in range(len(u_max_list)):
                filename = subfiles2[k]
                order = []
                data = np.load(osp.join('./data/results/128/',filename),allow_pickle=True).item()
                cont = data['cont']
                for j in range(len(cont)):
                    res = pnas_order(cont[j])
                    if not np.isnan(res):
                        order.append(res)
                    print(key,k,j)
                order = np.array(order)
                mean[k],std[k] = order.mean(),order.std()
            np.save(osp.join('./data/results/128/',key+' topology 30'),{'mean':mean,'std':std})
        # analyze_1()
        data = np.load(osp.join('./data/results/128/',key+' topology 20.npy'),allow_pickle=True).item()
        mean,std = data['mean'],data['std']
        # print(key,len(mean))
        mean = np.concatenate((np.array([no_noise[0,0]]),mean))
        std = np.concatenate((np.array([0.0]), std))
        ax0.plot(np.linspace(0,2.0,len(mean)) , mean, color=colors[3-i], marker='o',label=label_list[i])
        ax0.fill_between(np.linspace(0,2.0,len(mean)), mean - std**2, mean + std**2, color=colors[3-i], alpha=0.4)

    plt.legend(ncol=2,fontsize=ticksize,frameon=False,loc=1)
    plt.xticks([0,2],['0','2'],fontsize=ticksize)
    plt.yticks([0.5,1],['0.5','1'],fontsize=ticksize)
    plt.ylim(0.5,1)

    plt.xlabel(r'$\mathbf{\delta A}_{\max}$',fontsize=fontsize,labelpad=-20)
    plt.ylabel(r'$R_{150:200}$',fontsize=fontsize,labelpad=-25)
    plt.text(0.0,0.9,'(b)',fontsize=textsize)

    ax = fig.add_subplot(gs[4:8, 3:6]) ####################################################################################################################
    axins = ax.inset_axes(
        [15, 10, 45, 20], transform=ax.transData)
    subfiles3 = os.listdir('./data/topology')
    subfiles3 = [_ for _ in subfiles3 if '128' in _][::-1]
    for i in range(4):
        A = np.load(osp.join('./data/topology', subfiles3[i]))
        in_degree = A.sum(axis=1)
        result = np.unique(in_degree, return_counts=True)
        xs = np.linspace(0, np.max(result[0]), 1000)
        cs = CubicSpline(result[0], result[1])
        ys = cs(xs)
        ax.bar(*np.unique(in_degree, return_counts=True),alpha=0.5,color=colors[3-i])
        ax.plot(xs,ys,lw=2,color=colors[3-i])
        ax.set_ylim(0,35)
        ax.set_xlim(0,65)

        axins.bar(*np.unique(in_degree, return_counts=True),alpha=0.5,color=colors[3-i])
        axins.plot(xs, cs(xs), lw=2, color=colors[3-i])
        # axins.fill_between(xs, np.zeros_like(xs),cs(xs), color=colors[3-i], alpha=0.3)
        axins.set_xlim(0, 15)
        axins.set_ylim(0, 35)
        axins.set_xticks([])
        axins.set_yticks([])
    plt.text(3.0,30,'(c)',fontsize=textsize)
    xyA = (16.0, 10.5)
    xyB = (0, 0.0)
    coordsA = "data"
    coordsB = "data"
    con1 = ConnectionPatch(xyA, xyB,
                          coordsA, coordsB,
                          arrowstyle="-",
                          shrinkA=5, shrinkB=5,
                          mutation_scale=20,
                          fc="w",color='g')
    ax.add_artist(con1)
    xyA = (61.0, 10.0)
    xyB = (15.0, 0.0)
    coordsA = "data"
    coordsB = "data"
    con2 = ConnectionPatch(xyA, xyB,
                          coordsA, coordsB,
                          arrowstyle="-",
                          shrinkA=5, shrinkB=5,
                          mutation_scale=20,
                          fc="w",color='g')
    ax.add_artist(con2)

    ax.set_xticks([0,65])
    ax.set_xticklabels(['0','65'],fontsize=ticksize)
    ax.set_yticks([0,35])
    ax.set_yticklabels(['0','35'],fontsize=ticksize)
    ax.set_xlabel("In degree",fontsize=fontsize,labelpad=-15)
    ax.set_ylabel(r"$\#$ of Nodes",fontsize=fontsize,labelpad=-20)

    xyA = (14.0, 3.0)
    xyB = (1.0, 28.0)
    coordsA = "data"
    coordsB = "data"
    con3 = ConnectionPatch(xyA, xyB,
                          coordsA, coordsB,
                          arrowstyle="->",
                          shrinkA=5, shrinkB=5,
                          mutation_scale=40,
                          fc=[199/255,105/255,142/255],color='red',alpha=0.5)
    con3.set_linewidth(5)
    axins.add_artist(con3)

    colorss = [[204/255,77/255,27/255,1.0],[247/255,247/255,227/255,0.5],[0/255,114/255,192/255,1.0]]
    colorss = [[255/255,1/255,1/255,1.0],[247/255,247/255,227/255,0.5],[0/255,0/255,255/255,1.0]]
    # colorss = ["darkorange", "gold", "lawngreen", "lightseagreen"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colorss)
    nodes = [0., 0.5, 1.0]
    cmap2 = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colorss)))
    subfiles4 = [_ for _ in files if 'topology' in _]
    subfiles3 = subfiles3[::-1]
    index_list = ['(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)']
    for i in range(4):
        fig.add_subplot(gs[i*2:i*2+2,6:7]) ####################################################################################################################
        # W = np.load('./data/results/128/128 2137 baydry topology G_max=1.0.npy')
        W = np.load(osp.join('./data/results/128/',subfiles4[i]))
        A = np.load(osp.join('./data/topology/',subfiles3[i]))
        plt.imshow(W, extent=[0, 128, 0, 128], cmap=cmap2.reversed(), aspect='auto',vmax=0.1,vmin=-0.1)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(label_list[3-i],fontsize=fontsize,labelpad=0)
        plt.xlabel(index_list[i],fontsize=textsize)

        fig.add_subplot(gs[i*2:i*2+2,7:8]) ####################################################################################################################
        G = nx.from_numpy_array(W)
        edgeWeig = [G.edges[i]['weight'] for i in G.edges()]
        node_degree = [_[1]*0.3 for _ in G.degree()]
        options = {
            'node_size':node_degree,
            "node_color": [200/255,30/255,0/255], #"#A0CBE2",
            # "width": 2,
            'alpha':0.5,
            "edge_cmap": cmap2.reversed(),
            'edge_color': edgeWeig,
            'edge_vmin':-0.5,
            'edge_vmax':0.5,
            "with_labels": False,
        }
        nx.draw(G, pos=nx.spring_layout(G),**options)
        plt.text(0.5,-0.2,index_list[i+4],fontsize=textsize)

        # nx.draw(G, pos=nx.spring_layout(G),node_size=node_degree,node_color=node_degree,cmap=cmap2.reversed(),alpha=0.5,edge_color=edgeWeig,edge_cmap=cmap2.reversed())
    # plt.colorbar()
    plt.show()


def plot_v1():
    files = os.listdir('./data/results/128')

    fontsize = 18
    ticksize = 15
    textsize = 20
    seven_color = [[75 / 255, 102 / 255, 173 / 255], [98 / 255, 190 / 255, 166 / 255],
                   [205 / 255, 234 / 255, 157 / 255], 'gold',
                   [253 / 255, 186 / 255, 107 / 255], [235 / 255, 96 / 255, 70 / 255], [163 / 255, 6 / 255, 67 / 255]
                   ]
    rc_fonts = {
        "text.usetex": True,
        'text.latex.preview': True,  # Gives correct legend alignment.
        'mathtext.default': 'regular',
        'text.latex.preamble': [r"""\usepackage{bm}""", r"""\usepackage{amsmath}""", r"""\usepackage{amsfonts}"""],
        'font.sans-serif': 'Times New Roman'
    }
    import matplotlib
    matplotlib.rcParams.update(rc_fonts)

    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(left=0.05, bottom=0.07, right=0.96, top=0.97, hspace=0.5, wspace=0.7)
    gs = GridSpec(8, 8, figure=fig)

    fig.add_subplot(gs[0:4,0:4]) ####################################################################################################################
    def analyze_0():
        no_noise = []
        common_noise = []
        uncorr_noise = []
        subfiles = [_ for _ in files if 'u_max' in _]
        for filename in subfiles:
            sub_ord_cont,sub_ord_true = [],[]
            data = np.load(osp.join('./data/results/128/',filename),allow_pickle=True).item()
            cont,true = data['cont'],data['true']
            for k in range(len(cont)):
                sub_ord_cont.append(pnas_order(cont[k]))
                sub_ord_true.append(pnas_order(true[k]))
            if 'mode=0' in filename:
                common_noise.append(sub_ord_cont)
            if 'mode=1' in filename:
                uncorr_noise.append(sub_ord_cont)
            no_noise.append(sub_ord_true)

        no_noise = np.array(no_noise)
        common_noise = np.array(common_noise)
        uncorr_noise = np.array(uncorr_noise)
        no_noise = np.array([no_noise.mean(axis=1),no_noise.std(axis=1)])[:,:3]
        common_noise = np.array([common_noise.mean(axis=1),common_noise.std(axis=1)])
        uncorr_noise = np.array([uncorr_noise.mean(axis=1),uncorr_noise.std(axis=1)])
        np.save('./data/results/128/analyze_u',{'no':no_noise,'comm':common_noise,'unc':uncorr_noise})
    # analyze_0()
    data = np.load('./data/results/128/analyze_u.npy',allow_pickle=True).item()
    no_noise,common_noise,uncorr_noise = data['no'],data['comm'],data['unc']
    index = ['Low','Middle','High']
    bar_width = 0.2

    plt.bar(np.arange(len(index))+bar_width*0,no_noise[0] , width=bar_width, color=seven_color[0], alpha=0.6, lw=2.0,label='No Noise')
    plt.bar(np.arange(len(index))+bar_width*1,common_noise[0] , width=bar_width, color=seven_color[1], alpha=0.6, lw=2.0,label='Common')
    plt.bar(np.arange(len(index))+bar_width*2,uncorr_noise[0] , width=bar_width, color=seven_color[6], alpha=0.6, lw=2.0,label='Uncorrelated')
    for i in range(3):
        plt.errorbar((np.arange(len(index))+bar_width*0)[i:i+1],no_noise[0,i], no_noise[1,i], ecolor=seven_color[0],capsize=5)
        plt.errorbar((np.arange(len(index))+bar_width*1)[i:i+1],common_noise[0,i], common_noise[1,i], ecolor=seven_color[1],capsize=5)
        plt.errorbar((np.arange(len(index))+bar_width*2)[i:i+1],uncorr_noise[0,i], uncorr_noise[1,i], ecolor=seven_color[6],capsize=5)

    plt.xticks(np.arange(len(index))+bar_width,index,fontsize=ticksize)
    plt.yticks([0,1],['0','1'],fontsize=ticksize)
    plt.ylim(0,1.2)
    plt.legend(ncol=1,fontsize=ticksize,frameon=False,loc=2)
    plt.ylabel(r'$R_{150:200}$',fontsize=fontsize,labelpad=-13)
    plt.text(2.3,1.09,'(a)',fontsize=textsize)

    ax0 = fig.add_subplot(gs[0:2,4:8]) ####################################################################################################################
    xyA = (0.85,0.85)
    xyB = (0.85,0.65)
    coordsA = ax0.transData
    coordsB = ax0.transData
    con0 = ConnectionPatch(xyA, xyB,
                           coordsA, coordsB,
                           arrowstyle="->",
                           shrinkA=5, shrinkB=5,
                           mutation_scale=40,
                           fc=[199 / 255, 105 / 255, 142 / 255], color='red', alpha=0.5) #'salmon'
    con0.set_linewidth(5)
    ax0.add_artist(con0)
    subfiles1 = [_ for _ in files if 'G_max' in _]
    key_list = ['bay','ER','SF1','SF2'][::-1]
    label_list = ['Baydry','ER','SF1','SF2'][::-1]
    u_max_list = ['{:.1f}'.format(i*0.1+0.1) for i in range(30)]

    for i in range(4):
        key = key_list[i]
        subfiles2 = [_ for _ in subfiles1 if key in _]
        def analyze_1():
            mean,std = np.zeros(len(u_max_list)),np.zeros(len(u_max_list))
            for k in range(len(u_max_list)):
                filename = subfiles2[k]
                order = []
                data = np.load(osp.join('./data/results/128/',filename),allow_pickle=True).item()
                cont = data['cont']
                for j in range(len(cont)):
                    res = pnas_order(cont[j])
                    if not np.isnan(res):
                        order.append(res)
                    print(key,k,j)
                order = np.array(order)
                mean[k],std[k] = order.mean(),order.std()
            np.save(osp.join('./data/results/128/',key+' topology 30'),{'mean':mean,'std':std})
        # analyze_1()
        data = np.load(osp.join('./data/results/128/',key+' topology 20.npy'),allow_pickle=True).item()
        mean,std = data['mean'],data['std']
        # print(key,len(mean))
        mean = np.concatenate((np.array([no_noise[0,0]]),mean))
        std = np.concatenate((np.array([0.0]), std))
        ax0.plot(np.linspace(0,2.0,len(mean)) , mean, color=colors[3-i], marker='o',label=label_list[i])
        ax0.fill_between(np.linspace(0,2.0,len(mean)), mean - std**2, mean + std**2, color=colors[3-i], alpha=0.4)

    plt.legend(ncol=4,fontsize=ticksize,frameon=False,loc=1,bbox_to_anchor=[1.03,1.05],handlelength=1.0,handletextpad=0.5,columnspacing=0.8)
    plt.xticks([0,2],['0','2'],fontsize=ticksize)
    plt.yticks([0.5,1],['0.5','1'],fontsize=ticksize)
    plt.ylim(0.5,1)

    plt.xlabel(r'$\mathbf{\delta A}_{\max}$',fontsize=fontsize,labelpad=-15)
    plt.ylabel(r'$R_{150:200}$',fontsize=fontsize,labelpad=-25)
    plt.text(0.0,0.80,'(b)',fontsize=textsize)

    ax = fig.add_subplot(gs[2:4, 4:8]) ####################################################################################################################
    axins = ax.inset_axes(
        [15, 10, 45, 20], transform=ax.transData)
    subfiles3 = os.listdir('./data/topology')
    subfiles3 = [_ for _ in subfiles3 if '128' in _][::-1]
    for i in range(4):
        A = np.load(osp.join('./data/topology', subfiles3[i]))
        in_degree = A.sum(axis=1)
        result = np.unique(in_degree, return_counts=True)
        xs = np.linspace(0, np.max(result[0]), 1000)
        cs = CubicSpline(result[0], result[1])
        ys = cs(xs)
        ax.bar(*np.unique(in_degree, return_counts=True),alpha=0.5,color=colors[3-i])
        ax.plot(xs,ys,lw=2,color=colors[3-i])
        ax.set_ylim(0,35)
        ax.set_xlim(0,65)

        axins.bar(*np.unique(in_degree, return_counts=True),alpha=0.5,color=colors[3-i])
        axins.plot(xs, cs(xs), lw=2, color=colors[3-i])
        # axins.fill_between(xs, np.zeros_like(xs),cs(xs), color=colors[3-i], alpha=0.3)
        axins.set_xlim(0, 15)
        axins.set_ylim(0, 35)
        axins.set_xticks([])
        axins.set_yticks([])
    plt.text(3.0,27,'(c)',fontsize=textsize)
    xyA = (16.0, 10.5)
    xyB = (0, 0.0)
    coordsA = "data"
    coordsB = "data"
    con1 = ConnectionPatch(xyA, xyB,
                          coordsA, coordsB,
                          arrowstyle="-",
                          shrinkA=5, shrinkB=5,
                          mutation_scale=20,
                          fc="w",color='g')
    ax.add_artist(con1)
    xyA = (61.0, 10.0)
    xyB = (15.0, 0.0)
    coordsA = "data"
    coordsB = "data"
    con2 = ConnectionPatch(xyA, xyB,
                          coordsA, coordsB,
                          arrowstyle="-",
                          shrinkA=5, shrinkB=5,
                          mutation_scale=20,
                          fc="w",color='g')
    ax.add_artist(con2)

    ax.set_xticks([0,65])
    ax.set_xticklabels(['0','65'],fontsize=ticksize)
    ax.set_yticks([0,35])
    ax.set_yticklabels(['0','35'],fontsize=ticksize)
    ax.set_xlabel("In degree",fontsize=fontsize,labelpad=-15)
    ax.set_ylabel(r"$\#$ of Nodes",fontsize=fontsize,labelpad=-20)

    xyA = (14.0, 3.0)
    xyB = (1.0, 28.0)
    coordsA = "data"
    coordsB = "data"
    con3 = ConnectionPatch(xyA, xyB,
                          coordsA, coordsB,
                          arrowstyle="->",
                          shrinkA=5, shrinkB=5,
                          mutation_scale=40,
                          fc=[199/255,105/255,142/255],color='red',alpha=0.5)
    con3.set_linewidth(5)
    axins.add_artist(con3)

    colorss = [[204/255,77/255,27/255,1.0],[247/255,247/255,227/255,0.5],[0/255,114/255,192/255,1.0]]
    colorss = [[255/255,1/255,1/255,1.0],[247/255,247/255,227/255,0.5],[0/255,0/255,255/255,1.0]]
    # colorss = ["darkorange", "gold", "lawngreen", "lightseagreen"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colorss)
    nodes = [0., 0.5, 1.0]
    cmap2 = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colorss)))
    subfiles4 = [_ for _ in files if 'topology' in _]
    subfiles3 = subfiles3[::-1]
    index_list = ['(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)']
    np.random.seed(0)
    for i in range(4):
        fig.add_subplot(gs[4:6,i*2:i*2+2]) ####################################################################################################################
        # W = np.load('./data/results/128/128 2137 baydry topology G_max=1.0.npy')
        W = np.load(osp.join('./data/results/128/',subfiles4[i]))
        A = np.load(osp.join('./data/topology/',subfiles3[i]))
        plt.imshow(W, extent=[0, 128, 0, 128], cmap=cmap2.reversed(), aspect='auto',vmax=0.1,vmin=-0.1)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(label_list[3-i],fontsize=fontsize,labelpad=0)
        # plt.xlabel(index_list[i],fontsize=textsize)
        plt.text(5,10,index_list[i],fontsize=textsize)
        # plt.text(-2,-3,label_list[3-i],fontsize=fontsize,rotation=90)

        fig.add_subplot(gs[6:8,i*2:i*2+2]) ####################################################################################################################
        G = nx.from_numpy_array(W)
        edgeWeig = [G.edges[i]['weight'] for i in G.edges()]
        node_degree = [_[1]*0.3 for _ in G.degree()]
        options = {
            'node_size':node_degree,
            "node_color": [200/255,30/255,0/255], #"#A0CBE2",
            # "width": 2,
            'alpha':0.5,
            "edge_cmap": cmap2.reversed(),
            'edge_color': edgeWeig,
            'edge_vmin':-0.5,
            'edge_vmax':0.5,
            "with_labels": False,
        }
        nx.draw(G, pos=nx.spring_layout(G),**options)
        plt.text(0.5,-0.2,index_list[i+4],fontsize=textsize)
        # plt.text(-2,-3,label_list[3-i],fontsize=fontsize,rotation=90)
        # nx.draw(G, pos=nx.spring_layout(G),node_size=node_degree,node_color=node_degree,cmap=cmap2.reversed(),alpha=0.5,edge_color=edgeWeig,edge_cmap=cmap2.reversed())
    # plt.colorbar()
    plt.show()

# plot()
plot_v1()