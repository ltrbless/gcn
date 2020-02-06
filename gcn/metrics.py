import tensorflow as tf

# 注：loss的shape与mask的shape相同，等于样本的数量：(None,），所以 loss *= mask 是向量点乘(内积)。


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    # preds = preds[:140]
    # labels = labels[:140]
    # Epoch: 0001 train_loss= 1.95461 train_acc= 0.07857 val_loss= 0.00623 val_acc= 0.17600 time= 0.16556
    # Test set results: cost= 0.01496 accuracy= 0.53300 time= 0.03191
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)  # [1, 1, ... , 0, 0] 前面连续_len个为1
    # _len == 140 : 2708 / 140 = 19.342857 & mask = [19.342857 19.342857 19.342857 ...  0.        0.        0.      ]
    mask /= tf.reduce_mean(mask)  # 扩大了tf.reduce_mean(mask)倍，因此要除以这个数
    loss *= mask
    #  不扩大 2708 / 140 倍，那么loss的均值就太小了，导致无法在下降了
    #  加之后第一个loss 1.95461 不加第一个loss 0.10857
    #  由于不带标签数据的交叉熵为0(不带标签的，也就是mask=[False]的，人为掩盖的) ，
    #  其实乘以这个倍数之后就相当于没标签的交叉熵是按照带标签的交叉熵来代替，避免
    #  由于半监督学习时，有些数据无标签导致计算的损失值直接变得特别小，很难再调参，
    #  所以把那部分没标签得数据的损失值按照带标签的来代替（通过比例），大体这个意思
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
