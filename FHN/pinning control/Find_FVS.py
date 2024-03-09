import networkx as nx  # version: 2.8 --> 2.6.3,   scipy 1.5.0 -> 1.8.0
import numpy as np
# print(nx.__version__)

# def plot_G(network, name='G'):
#     import matplotlib.pyplot as plt
#     ps = nx.shell_layout(network)
#     #在1*3的画板中显示于第一格
#     plt.subplot(111)
#     plt.title(name)
#     nx.draw(network, ps, with_labels=True, node_color='r')
#     plt.show()


def cal_SCC(G):
    SCCs = sorted(nx.strongly_connected_components(G), key=len)
    for SCC in SCCs:
        end = '\n' if len(SCC)>1 else ''
        print(SCC, end=end)
    largest_SCC = sorted(list(SCC))
    print('largest_SCC:{}'.format(largest_SCC))

    for node in largest_SCC:
        reduced_G = G.copy()
        reduced_G.remove_nodes_from([node])
        flag = nx.is_directed_acyclic_graph(reduced_G)
        print('FVS={},  DAG:{}'.format(node, flag))


def zqx_multipartite_layout(G, subset_key="subset", align="vertical", scale=1, center=None, G_dim=2):
    """Position nodes in layers of straight lines.

    Parameters
    ----------
    G : NetworkX graph or list of nodes
        A position will be assigned to every node in G.

    subset_key : string (default='subset')
        Key of node data to be used as layer subset.

    align : string (default='vertical')
        The alignment of nodes. Vertical or horizontal.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    Examples
    --------
    >>> G = nx.complete_multipartite_graph(28, 16, 10)
    >>> pos = nx.multipartite_layout(G)

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    Network does not need to be a complete multipartite graph. As long as nodes
    have subset_key data, they will be placed in the corresponding layers.

    """
    import numpy as np
    from networkx.drawing.layout import rescale_layout, _process_params

    if align not in ("vertical", "horizontal"):
        msg = "align must be either vertical or horizontal."
        raise ValueError(msg)

    G, center = _process_params(G, center=center, dim=G_dim)
    if len(G) == 0:
        return {}

    layers = {}
    for v, data in G.nodes(data=True):
        try:
            layer = data[subset_key]
        except KeyError:
            msg = "all nodes must have subset_key (default='subset') as data"
            raise ValueError(msg)
        layers[layer] = [v] + layers.get(layer, [])

    # Sort by layer, if possible
    try:
        layers = sorted(layers.items())
    except TypeError:
        layers = list(layers.items())

    pos = None
    nodes = []
    width = len(layers)

    max_height = 0
    for i, (_, layer) in enumerate(layers):
        max_height = max(max_height, len(layer))

    last_max_x = 0
    for i, (_, layer) in enumerate(layers):
        height = len(layer)
        # r = 1
        # scale = height / max_height
        # thetas = np.linspace(np.pi/2 - np.pi/2 * scale, np.pi/2 + np.pi/2 * scale, height)
        # xs = last_max_x + r * np.sin(thetas)
        # ys = r * np.cos(thetas)
        # layer_pos = np.column_stack([xs, ys])
        # last_max_x = np.max(xs) + 0.1

        # r = i
        # thetas = np.linspace(0, np.pi * 2, height)
        # xs = r * np.sin(thetas)
        # ys = r * np.cos(thetas)
        # layer_pos = np.column_stack([xs, ys])
        # if G_dim==3:
        #     # layer_pos = np.column_stack([xs, ys, np.arange(height) / height])
        #     # layer_pos = np.column_stack([xs, ys, np.arange(height) * 0])
        #     layer_pos = np.column_stack([xs, ys, np.random.rand(height)])

        # multi-layer shown
        height = len(layer)
        xs = np.repeat(i, height)
        ys = np.arange(0, height, dtype=float)
        offset = ((width - 1) / 2, (height - 1) / 2)
        layer_pos = np.column_stack([xs, ys]) - offset

        if pos is None:
            pos = layer_pos
        else:
            pos = np.concatenate([pos, layer_pos])
        nodes.extend(layer)

    pos = rescale_layout(pos, scale=scale) + center
    if align == "horizontal":
        pos = pos[:, ::-1]  # swap x and y coords
    pos = dict(zip(nodes, pos))
    return pos

# https://networkx.org/documentation/latest/auto_examples/graph/plot_dag_layout.html#sphx-glr-auto-examples-graph-plot-dag-layout-py
def plot_G_DAG(A, FVS, with_labels=True):
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    reduced_G = G.copy()
    reduced_G.remove_nodes_from(FVS)
    flag = nx.is_directed_acyclic_graph(reduced_G)
    print('flag:{}'.format(flag))
    assert flag, 'DAG?'
    reduced_nodes = list(reduced_G.nodes())
    print('reduced_nodes:{}'.format(reduced_nodes))
    cal_SCC(G=G)


    align = 'horizontal'
    # align = 'vertical'
    for layer, nodes in enumerate(nx.topological_generations(reduced_G)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            if align == 'horizontal':
                # G.nodes[node]["layer"] = 8 - layer
                G.nodes[node]["layer"] = layer + 1
            else:
                G.nodes[node]["layer"] = layer + 1

    for node in FVS:
        if align == 'horizontal':
            # G.nodes[node]["layer"] = 9
            G.nodes[node]["layer"] = 0
        else:
            G.nodes[node]["layer"] = 0

    N = len(G)
    node_color = ['g'] * N
    print('nodes:{}'.format(N))
    largest_SCC = list(max(nx.strongly_connected_components(G), key=len))
    largest_SCC = sorted(largest_SCC)
    # for node in largest_SCC:
    for node in FVS:
        node_color[node] = 'purple'

    import matplotlib.pyplot as plt
    # Compute the multipartite_layout using the "layer" node attribute
    # pos = nx.multipartite_layout(G, subset_key="layer", align=align)
    G_dim = 2
    pos = zqx_multipartite_layout(G, subset_key="layer", align=align, G_dim=G_dim)

    fig = plt.figure(figsize=(3, 3))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.2, wspace=0.2)
    if G_dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        node_xyz = np.array([pos[v] for v in sorted(G)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
        # ax.scatter(*node_xyz.T, s=10, ec="w")
        x, y, z = node_xyz.T
        cmap = 'viridis'
        cmap = 'rainbow'
        ax.scatter(x, y, z, c=z, s=10)
        # ax.scatter(*node_xyz.T, s=10)
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray", lw=0.1)
    else:
        plt.subplot(111)
        # plt.title('DAG')
        # nx.draw(G, pos, with_labels=with_labels, node_color=node_color, node_size=5, font_size=5, width=0.1, alpha=0.2, arrowsize=5)
        node_size = 5
        arrowsize = 5
        nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, linewidths=0.1, alpha=1.0)
        nx.draw_networkx_edges(G, pos, edge_color='k', node_size=node_size, width=0.2, alpha=0.5, arrowsize=arrowsize)




    largest_SCC_G = G.subgraph(largest_SCC)
    pos = nx.shell_layout(largest_SCC_G)
    plt.figure(figsize=(6, 6))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.2, wspace=0.2)
    plt.subplot(111)
    # plt.title('DAG')
    nx.draw(largest_SCC_G, pos, with_labels=with_labels, node_color='g', node_size=500, font_size=10, width=1, alpha=0.2, arrowsize=10)
    # nx.draw_networkx_nodes(largest_SCC_G, pos, node_color='g', node_size=500, linewidths=1, alpha=0.5)
    # nx.draw_networkx_edges(largest_SCC_G, pos, edge_color='k', node_size=500, width=1, alpha=0.2, arrowsize=10)
    plt.show()

    # fig, ax = plt.subplots()
    # nx.draw_networkx(G, pos=pos, ax=ax)
    # ax.set_title("DAG layout in topological order")
    # fig.tight_layout()
    # plt.show()



    # cal_SCC(G=network)
    # N = len(network)
    # node_color = ['r'] * N
    # print('nodes:{}'.format(N))
    # largest_SCC = list(max(nx.strongly_connected_components(network), key=len))
    # largest_SCC = sorted(largest_SCC)
    # for node in largest_SCC:
    #     node_color[node] = 'g'
    #
    # print('largest_SCC:{}'.format(largest_SCC))
    #
    #
    # ps = nx.multipartite_layout(network)
    # # 在1*3的画板中显示于第一格
    # plt.subplot(111)
    # plt.title(name)
    # nx.draw(network, ps, with_labels=with_labels, node_color=node_color, node_size=50, font_size=5)
    # plt.show()


def plot_G(network, name='G', with_labels=True):
    cal_SCC(G=network)
    N = len(network)
    node_color = ['r'] * N
    print('nodes:{}'.format(N))
    largest_SCC = list(max(nx.strongly_connected_components(network), key=len))
    largest_SCC = sorted(largest_SCC)
    for node in largest_SCC:
        node_color[node] = 'g'

    print('largest_SCC:{}'.format(largest_SCC))
    import matplotlib.pyplot as plt
    # ps = nx.shell_layout(network)
    # ps = nx.spring_layout(network, seed=3068)
    # ps = nx.nx_agraph.graphviz_layout(network, prog="twopi", root=0)
    # ps = nx.kamada_kawai_layout(network)
    ps = nx.multipartite_layout(network)

    # ps = nx.random_layout(network, seed=3068)


    # ps = nx.spectral_layout(network)
    # ps = nx.spiral_layout(network)

    #在1*3的画板中显示于第一格
    plt.subplot(111)
    plt.title(name)
    nx.draw(network, ps, with_labels=with_labels, node_color=node_color, node_size=50, font_size=5)
    plt.show()

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
    # nx.from_numpy_matrix(A)  传入邻接矩阵
    # A[fr, to] = 1,  edge: fr -> to

    FVS = find_FVS(A=A)
    print('FVS:{}'.format(FVS))
    print('done!')
