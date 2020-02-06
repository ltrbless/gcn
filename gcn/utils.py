import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

# 在用python进行科学运算时，常常需要把一个稀疏的np.array压缩，
# 这时候就用到scipy库中的sparse.csr_matrix(csr:Compressed Sparse Row marix)
# 和sparse.csc_matric(csc:Compressed Sparse Column marix)
#  示例解读
# >>> indptr = np.array([0, 2, 3, 6])
# >>> indices = np.array([0, 2, 2, 0, 1, 2])
# >>> data = np.array([1, 2, 3, 4, 5, 6])
# >>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
# array([[1, 0, 2],
#        [0, 0, 3],
#        [4, 5, 6]])
# 按row行来压缩
# 对于第i行，非0数据列是indices[indptr[i]:indptr[i+1]] 数据是data[indptr[i]:indptr[i+1]]


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => 训练实例的特征向量，是scipy.sparse.csr.csr_matrix类对象，shape:(140, 1433)
    ind.dataset_str.tx => 测试实例的特征向量,shape:(1000, 1433)
    ind.dataset_str.allx => 有标签的+无无标签训练实例的特征向量，是ind.dataset_str.x的超集，shape:(1708, 1433)
    ind.dataset_str.y => 训练实例的标签，独热编码，numpy.ndarray类的实例，是numpy.ndarray对象，shape：(140, 7)
    ind.dataset_str.ty => 测试实例的标签，独热编码，numpy.ndarray类的实例,shape:(1000, 7)
    ind.dataset_str.ally => 对应于ind.dataset_str.allx的标签，独热编码,shape:(1708, 7)
    ind.dataset_str.graph => 图数据，collections.defaultdict类的实例，格式为 {index：[index_of_neighbor_nodes]}
    ind.dataset_str.test.index => 测试实例的id，# 1708 -  2707  1000个

    上述文件必须都用python的pickle模块存储

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):  # Python中版本获取Python版本号
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    # graph: defaultdict(<class 'list'>, {0: [633, 1862, 2582], 1: [2, 652, 654], 2: [1986, 332, 1666, 1, 1454],

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]  # 换一下 和test对应起来
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    _len = len(y)  # 　len(y)　１４０
    # _len == 140  Test set results: cost= 1.00631 accuracy= 0.80900 time= 0.03092
    # _len == 200  Test set results: cost= 0.91600 accuracy= 0.85800 time= 0.03096
    # _len == 500  Test set results: cost= 0.86395 accuracy= 0.85300 time= 0.03192
    # _len == 1000 Test set results: cost= 0.83154 accuracy= 0.86800 time= 0.03092
    idx_train = range(_len)
    idx_val = range(_len, _len + 500)
    idx_test = test_idx_range.tolist()
# 一共 2708 篇论文
# train_mask中的[0,140)范围的是True，其余是False；
# val_mask中范围为(140, 640]范围为True，其余的是False；
# test_mask中范围为[1708,2707]范围是True，其余的是False；
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        # print(type(mx))
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
#         print(mx.row, mx.col ) # mx就是特征矩阵， mx.row 就是行坐标  mx.y 就是列坐标
        coords = np.vstack((mx.row, mx.col)).transpose()  # transpose矩阵的转置
        # print(coords)  # (49216, 2)  49216  相当于边的个数
        values = mx.data
        shape = mx.shape
        return coords, values, shape  # coords 矩阵每个点的坐标 (49216, 2)  49216  相当于边的个数  values 矩阵归一化之后每个点的大小 shape 矩阵的形状

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    # print(rowsum) # (2708, 1) [[ 9.][23.][19.]...[18.][13.]]
    r_inv = np.power(rowsum, -1).flatten()  # 1 / x
#     print(r_inv) # [0.11111111 0.04347826 0.05263158 ... 0.05555556 0.07142857 0.07692308] (2708,)
#     print(r_inv.shape)
    r_inv[np.isinf(r_inv)] = 0.  # 如果刚刚发生除0错误，则置为 0
    r_mat_inv = sp.diags(r_inv)  # 构造为对角线
#     print(r_mat_inv)
#     t = sp.diags([1, 2, 3])
#     t.toarray()
#     array([[1., 0., 0.],
#            [0., 2., 0.],
#            [0., 0., 3.]])
    features = r_mat_inv.dot(features)  # 矩阵相乘
#     print(type(features))
#     写了这么多代码... 就是用广义度矩阵（一个点对应的一行的元素值(可能是特征值、出度/入度)，加起来作为这一个点的度）来进行归一化.
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

# feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})

    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})

    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
