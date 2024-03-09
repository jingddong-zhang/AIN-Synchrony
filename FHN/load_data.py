import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import os.path as osp
import os
import torch
import networkx as nx
# dict_data = scio.loadmat('./data/topology/PPI1.mat')
# A = dict_data['A']
# a = A.sum()
# np.save('./data/topology/PPI1.npy',A)


def data2matrix(name):
    dim,links = int(name.split()[0]),int(name.split()[1])
    A = np.zeros([dim,dim])
    path = osp.join('./data/foodweb', name)
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        index = data.index('*arcs\n')
        data = data[index + 1:index + 1 + links]
        for _ in data:
            left = _.split()
            left.pop()
            A[int(left[0])-1, int(left[1])-1] = 1.0
    np.save(osp.join('./data/topology',name.replace('txt','npy')),A)
    return A

def transform():
    files = os.listdir('./data/foodweb')
    for file in files:
        A = data2matrix(file)
        print(A.sum())
import re
# s = 'a\tb\tc\rd\re'
# s1 = re.sub('[\t\r]', '', s)

def electronic_circuit2matrix():
    with open('./data/s838.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
        '''
        get node, wirewise edge,flip flop wise edge pairs
        '''
        index1 = data.index('node;\n')
        index2 = data.index('edge;\n')
        index3 = data.index('flip-flops;\n')
        '''
        process node
        '''
        node_data = data[index1 + 1:index2]
        node_list = [_.split()[1][:-1] for _ in node_data]
        '''
        process wirewise edge
        '''
        edge_data = data[index2 + 1:index3]
        edge_list = [_.replace('assign ','')[:-2] for _ in edge_data]
        edge_list = [_.replace('&',' ') for _ in edge_list]
        edge_list = [re.sub('[=()|~]',' ',_) for _ in edge_list]
        edge_list = [_.split() for _ in edge_list]
        '''
        process flip flop edge
        '''
        ff_data = data[index3 + 1:]
        ff_data = ff_data[4:len(ff_data):5]
        ff_data = [re.sub('[<=;\n]','',_) for _ in ff_data]
        ff_data = [_.split() for _ in ff_data]
        '''
        get network topology
        '''
        num = len(node_list)
        A = np.zeros([num,num])
        for pair in edge_list:
            row_index = node_list.index(pair[0])
            for j in range(1,len(pair)):
                col_index = node_list.index(pair[j])
                A[row_index,col_index] = 1.0
        for pair in ff_data:
            row_index = node_list.index(pair[0])
            col_index = node_list.index(pair[1])
            A[row_index,col_index] = 1.0
    np.save('./data/topology/512 819 s838',A)


def generate_Di_ER():
    seed = 369
    k_mean = 2
    for N in [39,128,512]:
        A = nx.to_numpy_matrix(nx.erdos_renyi_graph(N,k_mean/(N-1),seed=seed,directed=True))
        np.save('./data/topology/{} Di_ER k={} seed={}'.format(N,k_mean,seed),A)


def Di_random_regular():
    A = nx.to_numpy_matrix(nx.random_regular_graph(k_mean, N, seed=seed))
    import scipy.sparse as sp
    B_data = sp.coo_matrix(A)
    B_weight = torch.Tensor(B_data.data).numpy()
    B = torch.Tensor(A)
    B = B.nonzero(as_tuple=False).t().contiguous().numpy().T.tolist()
    # print(A.shape,B.shape,B_weight.shape)
    # print(B[:10])
    delete_list = []
    for i in range(len(A)):
        ind1 = []
        current_list = [B[j] for j in range(i * 4, i * 4 + 4)]
        for j in range(4):
            # print(i,current_list[j],delete_list)
            if current_list[j] in delete_list:
                ind1.append(i * 4 + j)
        for j in range(4):
            if len(ind1) < 2:
                if not current_list[j] in delete_list:
                    ind1.append(i * 4 + j)
        del_ind = list(range(i * 4, i * 4 + 4))
        for _ in ind1:
            del_ind.remove(_)
        for _ in del_ind:
            delete_list.append(B[_][::-1])
        B_weight[ind1] = 0

    # print(type(np.array(B).T[0]))
    A = sp.coo_matrix((B_weight, (np.array(B).T[0], np.array(B).T[1])), shape=(N, N)).toarray()
    print(A.shape)


import numpy as np
def Di_SF(m,N,alpha_in,alpha_out,seed=36):
    '''
    :param m: mean degree
    :param N: num of nodes
    :param alpha_in: param related to in degree
    :param alpha_out: param related to out degree
    :param seed: random seed
    :return: connection matrix
    '''
    np.random.seed(seed)
    p_in = [np.power(i,-alpha_in) for i in range(1,N+1)]
    p_out = [np.power(i,-alpha_out) for i in range(1,N+1)]
    p_in = np.array(p_in)
    p_out = np.array(p_out)
    p_in *= 1/p_in.sum()
    p_out *= 1/p_out.sum()
    A = np.zeros([N,N])
    while A.sum()<m*N:
        in_index,out_index = np.where(np.random.multinomial(1,p_in)==1)[0],np.where(np.random.multinomial(1,p_out)==1)[0]
        A[in_index,out_index] = 1.0
    return A

def generate_Di_SF():
    k_mean = 4
    for N in [39,128,512]:
        gamma1,gamma2 = 2.5,2.5 # SF1
        # gamma1, gamma2 = 2.1, 2.9 # SF2
        if N == 128:
            seed = 3
        else:
            seed = 369
        A = Di_SF(k_mean,N,1/(gamma1-1),1/(gamma2-1),seed)
        np.save('./data/topology/{} Di_SF1 k={} gamma1={} gamma2={} seed={}'.format(N,k_mean,gamma1,gamma2,seed),A)

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


# generate_Di_SF()
# electronic_circuit2matrix()
# generate_Di_ER()