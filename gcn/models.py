from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()  # 获取当前类名的小写形式
        self.name = name  # 对于GCN来说是　　gcn

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []      # 保存每一个layer
        self.activations = []  # 保存每一次的输入，以及最后一层的输出

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):  # gcn
            self._build()

        # Build sequential layer model
        # 以一个两层GCN层为例，输入inputs是features
        self.activations.append(self.inputs)  # 初始化第一个元素为inputs，也就是features
        # 第一层，hidden=layer(self.activations[-1])，即hidden等于inputs的输出outputs，
        # 并将第一层的输出hidden=outputs加入到activations中
        # 同理，对第二层，hidden作为一个中间存储结果。最后activations分别存储了三个元素：第一层的输入，第二层的输入（第一层的输出），第二层的输出
        # 最后self.outputs=最后一层的输出
        for layer in self.layers:
            # convolve 卷积的实现。 返回卷积之后的结果。主要是根据论文中公式H^{(l+1)} = relu(\tilde{D}^{-1/2} \tilde{A}^{-1/2} \tilde{D}^{-1/2}  H^{(l)}W^{(l)})实现
            hidden = layer(self.activations[-1])  # Layer类重写了__call__ 函数，可以把对象当函数调用,__call__输入为inputs，输出为outputs
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        print('*' * 20)
        print(self.name)
        print(self.vars)
        print('-' * 20)

        # Build metrics
        self._loss()
        self._accuracy()
#     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(self.layers[0].vars.values())
#     self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
#                                               self.placeholders['labels_mask'])
#     self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
#                                     self.placeholders['labels_mask'])


        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim  # 1433 列数
        # self.input_dim = self.inputs.get_shape().as_list()[1]
        #  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]  # 7
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss # 正则化项  对权重进行限制  防止过拟合
        for var in self.layers[0].vars.values():
            print('2' * 20)
            print(var)
            print('2' * 20)
            # <tf.Variable 'gcn/graphconvolution_1_vars/weights_0:0' shape=(1433, 16) dtype=float32_ref>

            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error # 交叉熵损失函数
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,  # 1433
                                            output_dim=FLAGS.hidden1,  # 16
                                            placeholders=self.placeholders,  # 预定义数据格式
                                            # placeholders = {
                                            # #由于邻接矩阵是稀疏的，并且用LIL格式表示，因此定义为一个tf.sparse_placeholder(tf.float32)，可以节省内存
                                            #     'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
                                            #     # features也是稀疏矩阵，也用LIL格式表示，因此定义为tf.sparse_placeholder(tf.float32)，维度(2708, 1433)
                                            #     # print(features[2])
                                            #     # (2708, 1433)
                                            #     'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
                                            #     # print(y_train.shape[1])
                                            #     # 7
                                            #     'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
                                            #     'labels_mask': tf.placeholder(tf.int32),
                                            #     'dropout': tf.placeholder_with_default(0., shape=()),
                                            #     'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
                                            # }
                                            act=tf.nn.relu,  # 使用relu激活函数
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,  # 16
                                            output_dim=self.output_dim,  # 7
                                            placeholders=self.placeholders,
                                            act=lambda x: x,  # 无激活函数
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)  # 使用 softmax 函数进行分类


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,  # 1433
                                 output_dim=FLAGS.hidden1,  # 16
                                 placeholders=self.placeholders,  # 　预定义数据格式
                                 # placeholders = {
                                 # #由于邻接矩阵是稀疏的，并且用LIL格式表示，因此定义为一个tf.sparse_placeholder(tf.float32)，可以节省内存
                                 #     'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
                                 #     # features也是稀疏矩阵，也用LIL格式表示，因此定义为tf.sparse_placeholder(tf.float32)，维度(2708, 1433)
                                 #     # print(features[2])
                                 #     # (2708, 1433)
                                 #     'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
                                 #     # print(y_train.shape[1])
                                 #     # 7
                                 #     'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
                                 #     'labels_mask': tf.placeholder(tf.int32),
                                 #     'dropout': tf.placeholder_with_default(0., shape=()),
                                 #     'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
                                 # }
                                 act=tf.nn.relu,  # 使用relu激活函数
                                 dropout=True,  # 是否采用 dropout 以避免过拟合
                                 sparse_inputs=True,  # 稀疏矩阵输入
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,  # 16
                                 output_dim=self.output_dim,  # 7 哪一篇论文
                                 placeholders=self.placeholders,
                                 act=lambda x: x,  # 无激活函数
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)  # 　使用ｓｏｆｔｍａｘ进行分类
