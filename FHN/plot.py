import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import os.path as osp
import os
import torch
import networkx as nx
import re

def pnas_order(data, T=15000, delta=0.1):
    L = int(data.shape[1] / 2)
    data = data[T:, 0:L]
    mean1 = np.mean(data, axis=1)
    std1 = np.std(mean1)
    std2 = np.mean(np.std(data, axis=0))
    order = std1 / std2
    return order

def success_rate(data,thres=0.05):
    m = data.shape[0]
    T = 2000
    dim = int(data.shape[-1]/2)
    succ_rate_list = []
    for i in range(m):
        u = data[i,-T:,:dim]
        du = u-np.repeat(u[:,0:1],dim,axis=1)
        du_module = np.sqrt(du**2)
        mse_mean = np.mean(du_module,axis=0)
        succ_index = np.where(mse_mean<thres)[0]
        succ_rate = len(succ_index)/dim
        succ_rate_list.append(succ_rate)
    return succ_rate_list

# data = np.load('./data/results/39 Di_ER k=4 seed=369 mode=1 u_max=1.0 common.npy')
# s = success_rate(data)
files = os.listdir('./data/results/128')

def get_succ_rate(network):
    succ_rate_list = []
    for mode in ['mode={}'.format(i + 1) for i in range(4)]:
        sub_list = []
        for file in files:
            if network in file:
                if mode in file:
                    data = np.load(osp.join('./data/results/39',file))
                    succ_rate = success_rate(data)
                    # print(np.array(succ_rate).std())
                    sub_list.append([np.array(succ_rate).mean(),np.array(succ_rate).std()])
        succ_rate_list.append(np.array(sub_list))
    return np.array(succ_rate_list)

fontsize = 18
ticksize = 15
seven_color = [[75 / 255, 102 / 255, 173 / 255], [98 / 255, 190 / 255, 166 / 255],
               [205 / 255, 234 / 255, 157 / 255], 'gold',
               [253 / 255, 186 / 255, 107 / 255], [235 / 255, 96 / 255, 70 / 255], [163 / 255, 6 / 255, 67 / 255]
               ]
index = ['Ches','ER','SF1','SF2']
# tick_index = index
# tick_index[0] = 'Net'
# def summarize_data():
#     index = ['Chesapeake', 'ER', 'SF1', 'SF2']
#     all_succ_data = np.zeros([4,4,10,2]) # network,mode,u_max
#     for i in range(4):
#         succ = get_succ_rate(index[i])
#         all_succ_data[i,:] = succ
#     np.save('./data/results/39/summary',all_succ_data)

fig = plt.figure(figsize=(15, 6))
plt.subplots_adjust(left=0.07, bottom=0.10, right=0.92, top=0.9, hspace=0.2, wspace=0.7)

plt.subplot(231)
all_succ_data = np.load('./data/results/39/summary.npy')
print(all_succ_data.shape)
bar_width = 0.2
for i in range(4):
    x = np.arange(len(index))+bar_width*i
    mean = np.mean(all_succ_data[:, i, :,0], axis=1)
    std = np.std(all_succ_data[:, i, :, 0], axis=1)
    plt.bar(x,mean , width=bar_width, color=seven_color[i], alpha=0.6, lw=2.0, edgecolor=seven_color[i],label=f'mode{i+1}')
    for j in range(4):
        plt.errorbar(x[j:j+1],mean[j], std[j], ecolor=seven_color[i],capsize=10)
        pos = x[j:j+1]
        plt.scatter(np.random.uniform(-bar_width/3 + pos, bar_width/3 + pos, 10), all_succ_data[j,i,:,0], color=seven_color[i])


# for i in range(7):
#     plt.errorbar(index[i:i + 1], np.mean(succ_rate, axis=1)[i], np.std(succ_rate, axis=1)[i], ecolor=seven_color[i],
#                  capsize=6)
#     plt.scatter(np.random.uniform(-0.2 + 1.0 * i, 0.2 + 1.0 * i, 6), succ_rate[i], color=seven_color[i])
plt.yticks([0, 1], ['0', r'100$\%$'], fontsize=ticksize)
plt.xticks(np.arange(len(index))+ bar_width*1.5 , index,fontsize=ticksize,rotation=0)
plt.ylabel('Success Rate', fontsize=fontsize, labelpad=-40)
plt.legend(fontsize=ticksize,ncol=4,frameon=False)


plt.subplot(232)
data = np.load('./data/results/39/39 Di_SF2 k=4 gamma1=2.1 gamma2=2.9 mode=4 u_max=1.0 common.npy')
print(data.shape)
plt.imshow(data[0,-10000:, 0:39].T, extent=[0, 300, 0, 39], cmap='RdBu', aspect='auto')
plt.show()


