import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import os.path as osp
import os
import torch
import networkx as nx


A = np.load('./data/topology/39 117 Chesapeake.npy')
# A = np.load('./data/topology/128 2137 baydry.npy')
# A = np.load('./data/topology/PPI1.npy')
# A = np.load('./data/topology/512 819 s838.npy')
# N = 39
# k_mean = 4
# seed = 36



# print(np.abs(A-A.T).sum())
# print(np.where(A==1)[0])
# plt.imshow(A,cmap=plt.cm.hot_r, vmin=0, vmax=1)
outdegree = A.sum(axis=0)
indegree = A.sum(axis=1)
top_nodes = np.where(indegree==0)[0]
print(top_nodes,outdegree[top_nodes])
node1 = np.argmax(outdegree)
sub_node1 = np.where(A[:,38]==1)[0]
# print(sub_node1,np.sort(outdegree)[::-1][:5])

G = nx.from_numpy_array(A)
degree = nx.degree_histogram(G)
# print(degree)

x = range(len(degree))  # 生成X轴序列，从1到最大度
y = [z / float(sum(degree)) for z in degree]  # 将频次转化为频率
# plt.loglog(x, y, '.')
# plt.bar(x, y, width=0.5, color="blue")

# plt.subplot(211)
# plt.bar(np.arange(len(A)),outdegree)
# plt.subplot(212)
# plt.bar(np.arange(len(A)),indegree)
def degree_plot(degree):
    max_ = int(np.max(degree))
    distribution = np.zeros([max_+1])
    for i in range(max_+1):
        sub = np.where(degree==i)[0]
        distribution[i] = len(sub)
    return distribution
out_dis = degree_plot(outdegree[0])
in_dis = degree_plot(indegree.T[0])
# print(len(outdegree),len(out_dis))
plt.subplot(121)
plt.bar(range(len(out_dis)), out_dis, width=0.1, color="blue")
# plt.ylim(-0.1,20)
plt.title('{}'.format(outdegree.mean()))
plt.subplot(122)
plt.bar(range(len(in_dis)), in_dis, width=0.1, color="red")
# plt.ylim(-0.1,20)
plt.title('{}'.format(indegree.mean()))

G=nx.from_numpy_array(A)
ps=nx.circular_layout(G)#布置框架
# nx.draw(G,ps,with_labels=False,node_size=30)
# plt.bar(np.arange(len(A)),indegree)
plt.show()
