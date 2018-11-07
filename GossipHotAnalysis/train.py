# !/usr/bin/env python3

import tensorflow as tf
import numpy as np

import os
import sys

sys.path.append('./ref/bert/')
sys.path.append('./')

from model import *
from data import *

def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        # pylint: disable=W0212
        session = session._sess
    return session

def train():
    opts = parse_args()
    extractor = GossipCommentNumberExtractor()
    vocab = vocab_160k.Vocabulary('./corpus/w2v/v160k_big_string.txt')
    ds = GossipCommentNumberDataset('./dataset/use_data/*.txt', vocab, extractor)
    opts.vocab_size=vocab.size
    # wordsEmbeddings = tf.Variable(vocab.emb, dtype=tf.float32)
    # ds = ResumeSummaryDataset(filepattern, vocab, extractor)
    ckpt_dir = opts.ckpt_dir
    best_acc = 0
    with tf.Graph().as_default():
        with tf.device('/device:GPU:2'):
            wordsEmbeddings = tf.Variable(vocab.emb, dtype=tf.float32)
            global_step = tf.train.get_or_create_global_step()
            model = GossipCommentNumModel(wordsEmbeddings, opts, True, global_step)
            loss_op = model.loss
            train_op = model.train_op
            saver = tf.train.Saver()
            X = ds.iter_batches(opts.batch_size,
                                opts.max_tokens_per_sent)
            iter_num = 0
            with tf.train.MonitoredTrainingSession(checkpoint_dir=ckpt_dir,
                                                   hooks=[tf.train.StopAtStepHook(last_step=opts.iter_num),
                                                          tf.train.NanTensorHook(loss_op)],
                                                   config=tf.ConfigProto(
                                                       allow_soft_placement=True, log_device_placement=True)
                                                   ) as sess:
                try:
                    while not sess.should_stop():
                        Y = next(X)
                        # resume_tensor = tf.convert_to_tensor(Y[1], dtype=tf.int32)
                        # summary_tensor = tf.convert_to_tensor(Y[0], dtype=tf.int32)
                        train_val, loss_val, global_step_val, preds = sess.run([
                            train_op, loss_op, global_step, model.preds
                        ],
                            feed_dict={model.title_tokens:Y[0], model.scores:Y[1], model.lr:1e-4})
                        if iter_num%100 == 0:
                            print('[{}]\tloss:{:.4f}'.format(iter_num, loss_val))
                        iter_num += 1
                        acc_val = 1 - loss_val
                        if acc_val > best_acc:
                            best_acc = acc_val
                            print('Current best accuracy is: {}'.format(best_acc))
                            saver.save(get_session(sess), os.path.join(ckpt_dir, 'best_model'))
                except Exception as e:
                    print(e)
                    saver.save(get_session(sess), os.path.join(ckpt_dir, 'final_model'))

if __name__ == '__main__':
    train()