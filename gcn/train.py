from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed' 可用这三种数据集
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense' # 三种方式
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')  # 默认学习率
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')  # 迭代次数
# 第一层
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')  # 第一层隐藏层的单元个数
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')  # 保留 50% 神经元，防止过拟合

# 权值衰减：防止过拟合
# loss计算方式（权值衰减+正则化）：self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
# tf.nn.l2_loss(var) \sum(var**2)/2
# t = tf.nn.l2_loss([1., 2., 3.])
# sess = tf.Session()
# sess.run(t)  # ( 1**2 + 2**2 + 3**2 )/ 2 = 14/2 = 7
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
# 利用Chebyshev多项式递归计算卷积核  # K阶的切比雪夫近似矩阵的参数k
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
# cora 数据集简介：
# Cora数据集由机器学习论文组成，是近年来图深度学习很喜欢使用的数据集。在数据集中，论文分为以下七类之一:
# 1. 基于案例
# 2. 遗传算法
# 3. 神经网络
# 4. 概率方法
# 5. 强化学习
# 6. 规则学习
# 7. 理论
# 论文的选择方式是，在最终语料库中，每篇论文引用或被至少一篇其他论文引用。整个语料库中有2708篇论文。
# 在词干堵塞和去除词尾后，只剩下1433个独特的单词。文档频率小于10的所有单词都被删除。

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
# adj(邻接矩阵)：由于比较稀疏，邻接矩阵格式是LIL的，并且shape为(2708, 2708)
# features（特征矩阵）：每个节点的特征向量也是稀疏的，features.shape: (2708, 1433) 1433个词，有为1，没有为2.
# train_mask中的[0,140)范围的是True，其余是False；
# val_mask中范围为(140, 640]范围为True，其余的是False；
# test_mask中范围为[1708,2707]范围是True，其余的是False；
features = preprocess_features(features)  # 归一化特征矩阵
# preprocess_features()  --> return coords, values, shape
# coords 矩阵每个点的坐标 (49216, 2)  49216  相当于边的个数
# values 矩阵归一化之后每个点的大小
# shape 矩阵的形状 (2708, 1433)

if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]  # 把邻接矩阵处理为  \tilde{D}^{-1/2} \tilde{A}^{-1/2} \tilde{D}^{-1/2}
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':  # 切比雪夫近似，加速
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':  # Dense层就是所谓的全连接神经网络层
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    # 由于邻接矩阵是稀疏的，并且用LIL格式表示，因此定义为一个tf.sparse_placeholder(tf.float32)，可以节省内存
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    # features也是稀疏矩阵，也用LIL格式表示，因此定义为tf.sparse_placeholder(tf.float32)，维度(2708, 1433)
    # print(features[2])
    # (2708, 1433)
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    # print(y_train.shape[1])
    # 7
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
# # print(features[2][1])
# 1433
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(3):  # range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
