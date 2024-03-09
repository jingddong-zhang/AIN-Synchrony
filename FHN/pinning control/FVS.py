import networkx as nx  # version: 2.8 --> 2.6.3,   scipy 1.5.0 -> 1.8.0
import numpy as np


def heuristic_sort_SCC(G, largest_SCC, method='Multiply'):
    largest_SCC = list(largest_SCC)
    # self_loop_nodes = G.nodes_with_selfloops()
    self_loop_nodes = list(nx.nodes_with_selfloops(G))
    self_loop_flags = [node in self_loop_nodes for node in largest_SCC]
    if method == 'Multiply':
        heuristic_costs = [G.in_degree(node) * G.out_degree(node)
                           for node in largest_SCC]
    elif method == 'Addition':
        heuristic_costs = [G.in_degree(node) + G.out_degree(node)
                           for node in largest_SCC]
    elif method == 'Maximum':
        heuristic_costs = [max(G.in_degree(node), G.out_degree(node))
                           for node in largest_SCC]
    values = [(largest_SCC[i], -self_loop_flags[i], -heuristic_costs[i])
              for i in range(len(largest_SCC))]
    dtype = [('largest_SCC', np.int32),
             ('self_loop_falgs', np.int32),
             ('heuristic_costs', np.int32)]

    Res = np.array(values, dtype=dtype)
    Res.sort(order=['self_loop_falgs', 'heuristic_costs'])

    if len(Res) > 10 and Res[0] == Res[1]:
        print('Equal ###########################################')
    return Res[0][0]


def Fully_reduction_Graph(GG):
    self_loop = []
    G = GG.copy()
    # Step 1": if out(i)=0 or in(i)=0
    nodes = G.nodes()
    for node in nodes:
        if G.in_degree(node) == 0 or G.out_degree(node) == 0:
            G.remove_node(node)
    # Step 1": if out(i)=0 or in(i)=0
    nodes = G.nodes()
    nodes = list(nodes)
    for node in nodes:
        if G.in_degree(node) != 1:
            continue
        pre_node = list(G.predecessors(node))[0]
        if pre_node == node:
            continue
        for out_node in G.successors(node):
            G.add_edge(pre_node, out_node)
        G.remove_nodes_from([node])

    nodes = G.nodes()
    nodes = list(nodes)
    for node in nodes:
        if G.out_degree(node) != 1:
            continue
        suc_node = list(G.successors(node))[0]
        if suc_node == node:
            continue
        for in_node in G.predecessors(node):
            G.add_edge(in_node, suc_node)
        G.remove_nodes_from([node])
    return G

def is_reduant(A, FVS):
    print('Starting is_reduant function!')
    G = nx.from_numpy_matrix(A.transpose(), create_using=nx.DiGraph())
    copy_G = G.copy()
    copy_G.remove_nodes_from(FVS)
    assert nx.is_directed_acyclic_graph(copy_G), '{} is not a FVS'.format(FVS)
    print('Is FVS: ', nx.is_directed_acyclic_graph(copy_G))
    print('FVS number=', len(FVS))
    flag = 0
    for node in FVS:
        copy_G = G.copy()
        if len(FVS) > 1:
            tmp = []
            for j in FVS:
                if j == node:
                    continue
                tmp.append(j)
            # tmp = np.array(tmp)
            copy_G.remove_nodes_from(tmp)
        if nx.is_directed_acyclic_graph(copy_G):
            flag = 1
            FVS = tmp
            print('Reduant node %d' % node)
            print('### new FVS:', FVS)
    if flag == 0:
        print('No Reduant nodes!')
    return flag, FVS

def find_FVS(A, Fully_reduction=True, reduction=True, method='Multiply'):
    print('#' * 100)
    print('Begin of find_FVS')
    N = len(A)
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    # plot_G(network=G)
    print('Initial graph is a DAG: ', nx.is_directed_acyclic_graph(G))
    FVS = []
    while True:
        largest_SCC = list(max(nx.strongly_connected_components(G), key=len))
        if len(largest_SCC) == 1:
            reduced_G = G.subgraph(largest_SCC)
            print('SCC nodes', reduced_G.in_degree(largest_SCC))
            if reduced_G.in_degree(largest_SCC[0]) != 2:
                if len(reduced_G.nodes()) == 1:
                    break
                continue
        if Fully_reduction:
            reduced_G = G.subgraph(largest_SCC)
            reduced_G = Fully_reduction_Graph(reduced_G)
            largest_SCC = reduced_G.nodes()
        elif reduction:
            reduced_G = G.subgraph(largest_SCC)
        else:
            reduced_G = G.copy()

        rem_i = heuristic_sort_SCC(reduced_G, largest_SCC, method)

        G.remove_nodes_from([rem_i])
        flag = nx.is_directed_acyclic_graph(G)
        FVS.append(rem_i)
        print('去掉节点', rem_i)
        if flag:
            break
    while True: # remove the reduanat node!
        flag, FVS = is_reduant(A, FVS)
        if not flag:
            break
    FVS = sorted(FVS)
    print('End of find_FVS')
    print('%' * 100)



    return FVS


if __name__ == '__main__':
    A = np.array(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
    )

    A = np.load('./data/topology/39 117 Chesapeake.npy')
    A = np.load('./data/topology/128 2137 baydry.npy')
    # A = np.load('./data/topology/512 819 s838.npy')
    # nx.from_numpy_matrix(A)  传入邻接矩阵
    A = nx.to_numpy_matrix(nx.erdos_renyi_graph(100,0.02,directed=True))

    from load_data import Di_SF
    A = Di_SF(4,39,1/(2.5-1),1/(2.5-1),369)
    # A[fr, to] = 1,  edge: fr -> to
    A = np.load('./data/topology/39 Di_ER k=4 seed=369.npy')
    FVS = find_FVS(A=A)
    print('FVS:{}'.format(FVS))
    print('done!')
    indegree = A.sum(axis=1)
    print(np.where(indegree==0)[0])
