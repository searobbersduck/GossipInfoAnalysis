# !/usr/bin/env python3

import os
from glob import glob
import sys
sys.path.append('./ref/bert/')
sys.path.append('./')

import tensorflow as tf

from transformer import *
from params import *

def focal_loss(labels, logits, gamma=2.0, alpha=4.0):
    """
    focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: logits is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022
    :param labels: ground truth labels, shape of [batch_size]
    :param logits: model's output, shape of [batch_size, num_cls]
    :param gamma:
    :param alpha:
    :return: shape of [batch_size]
    """
    epsilon = 1.e-9
    labels = tf.to_int64(labels)
    labels = tf.convert_to_tensor(labels, tf.int64)
    logits = tf.convert_to_tensor(logits, tf.float32)
    num_cls = logits.shape[1]

    model_out = tf.add(logits, epsilon)
    onehot_labels = tf.one_hot(labels, num_cls)
    ce = tf.multiply(onehot_labels, -tf.log(model_out))
    weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
    return reduced_fl

def get_mask(inputs):
    # inputs:(N, T, C)
    mask = tf.sign(tf.abs(tf.reduce_sum(inputs, axis=-1)))
    mask = tf.ones_like(mask)
    # mask = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, inputs.get_shape().as_list()[-1]])
    return mask

class GossipCommentNumModel(object):
    def __init__(self, wordsEmbeddings, opts, is_training, num_class=2, global_step=None):
        # ops
        batch_size = opts.batch_size
        max_tokens_per_sent = opts.max_tokens_per_sent
        positional_enc_dim = opts.positional_enc_dim
        dropout_rate = opts.dropout_rate
        num_heads = opts.num_heads
        num_layers = opts.num_layers
        hidden_size = opts.hidden_size

        self.num_class = num_class
        self.lr = tf.placeholder(name='lr', shape=None, dtype=tf.float32)
        self.title_tokens = tf.placeholder(name='titles', shape=[batch_size, max_tokens_per_sent], dtype=tf.int32)
        self.scores = tf.placeholder(name='scores', shape=[batch_size], dtype=tf.int32)
        self.words_embeddings = tf.convert_to_tensor(wordsEmbeddings)
        self.title_tokens_emb = tf.nn.embedding_lookup(
            self.words_embeddings, self.title_tokens
        )
        with tf.variable_scope('encoder'):
            self.enc = self.title_tokens_emb
            self.positional_enc = positional_encoding(
                self.enc,
                positional_enc_dim,
                scope='resume_positional_enc'
            )
            enc_mask = get_mask(self.enc)
            enc_mask = tf.tile(tf.expand_dims(enc_mask, axis=2), [1, 1, self.positional_enc.get_shape().as_list()[-1]])
            self.enc = tf.concat([self.enc, self.positional_enc * enc_mask], axis=-1)
            self.enc = tf.layers.dropout(self.enc,
                                         rate=dropout_rate,
                                         training=tf.convert_to_tensor(is_training))
            for i in range(num_layers):
                with tf.variable_scope('num_block_{}'.format(i)):
                    self.enc = multihead_attention(
                        self.enc,
                        self.enc,
                        num_units=hidden_size,
                        num_heads=num_heads,
                        dropout_rate=dropout_rate,
                        is_training=is_training,
                        causality=False
                    )
                    self.enc = feedforward(
                        self.enc, num_units=[hidden_size*4, hidden_size]
                    )
        self.enc = tf.layers.conv1d(
            inputs=self.enc,
            filters=1,
            kernel_size=1,
            activation=tf.nn.relu,
            use_bias=True
        )
        self.enc = tf.squeeze(self.enc, axis=-1)
        self.w = tf.get_variable('w', shape=[max_tokens_per_sent, self.num_class], initializer=tf.contrib.layers.xavier_initializer())
        self.b = tf.get_variable('b', shape=[self.num_class], initializer=tf.contrib.layers.xavier_initializer())
        self.enc = tf.nn.xw_plus_b(self.enc, self.w, self.b)
        # self.enc = tf.squeeze(self.enc, axis=-1)
        # self.preds = tf.sigmoid(self.enc)
        self.enc = tf.nn.softmax(self.enc)
        self.preds = tf.cast(tf.argmax(self.enc, axis=1), tf.int32)
        self.acc = tf.reduce_sum(
            tf.cast(tf.equal(self.preds, self.scores), tf.int32))/opts.batch_size
        # self.loss = tf.losses.mean_squared_error(self.scores*100, self.preds)
        # self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     logits=self.enc,
        #     labels=self.scores
        # ))
        self.loss = tf.reduce_mean(focal_loss(self.scores, self.enc))
        # self.loss = tf.reduce_sum(self.loss)
        # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # self.loss += reg_losses
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients = [
            None if gradient is None else tf.clip_by_norm(gradient, 1.0)
            for gradient in gradients
        ]
        self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

def test_GossipCommentNumModel():
    vocab_size = 8000
    hidden_size = 256
    wordsEmbeddings = tf.get_variable('wordsEmbeddings', shape=[vocab_size, hidden_size], dtype=tf.float32)
    opts = parse_args()
    model = GossipCommentNumModel(wordsEmbeddings, opts, True)
    print('hello world!')

if __name__ == '__main__':
    test_GossipCommentNumModel()
