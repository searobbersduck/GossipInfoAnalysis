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

def val(sess, vocab, extractor, model, opts,
        cur_accuray, iter_num, ckpt_dir, saver):
    ds = GossipCommentNumberDataset('./dataset/use_data/*_val.txt',
                                    vocab, extractor,
                                    test=True)
    preds_stat = np.array([], dtype=np.int32)
    gt_stat = np.array([], dtype=np.int32)
    loss_stat = np.array([], dtype=np.float32)
    X = ds.iter_batches(opts.batch_size,
                        opts.max_tokens_per_sent)
    while True:
        try:
            Y = next(X)
            loss_val, preds, acc_val = sess.run([
                model.loss, model.preds, model.acc
            ],
                feed_dict={model.title_tokens: Y[0], model.scores: Y[2], model.lr: opts.lr})
            preds_stat = np.concatenate((preds_stat, preds))
            gt_stat = np.concatenate((gt_stat, Y[2]))
            loss_stat = np.append(loss_stat, loss_val)
        except:
            acc = np.sum(np.equal(preds_stat, gt_stat)) / preds_stat.size
            loss = np.sum(loss_stat) / loss_stat.size
            print('[{}]\tloss:{}\tacc:{:.4f}'.format(iter_num, loss, acc))
            print('pred label 1 number is: {}'.format(np.sum(np.equal(preds_stat, 1))))
            print('pred label 0 number is: {}'.format(np.sum(np.equal(preds_stat, 0))))
            print('gt label 1 number is: {}'.format(np.sum(np.equal(gt_stat, 1))))
            print('gt label 0 number is: {}'.format(np.sum(np.equal(gt_stat, 0))))
            if acc > cur_accuray:
                cur_accuray = acc
                print('Current best validation accuracy is: {}'.format(cur_accuray))
                saver.save(get_session(sess), os.path.join(ckpt_dir, 'best_model_{}'.format(iter_num)))
            break
    return cur_accuray


def train():
    opts = parse_args()
    extractor = GossipCommentNumberExtractor()
    vocab = vocab_160k.Vocabulary('./corpus/w2v/v160k_big_string.txt')
    ds = GossipCommentNumberDataset('./dataset/use_data/*_train.txt', vocab, extractor, shuffle_on_load=True)
    opts.vocab_size=vocab.size
    # wordsEmbeddings = tf.Variable(vocab.emb, dtype=tf.float32)
    # ds = ResumeSummaryDataset(filepattern, vocab, extractor)
    ckpt_dir = opts.ckpt_dir
    best_acc = 0
    preds_stat = np.array([], dtype=np.int32)
    gt_stat = np.array([], dtype=np.int32)
    loss_stat = np.array([], dtype=np.float32)
    with tf.Graph().as_default():
        with tf.device('/device:GPU:2'):
            wordsEmbeddings = tf.Variable(vocab.emb, dtype=tf.float32)
            global_step = tf.train.get_or_create_global_step()
            model = GossipCommentNumModel(wordsEmbeddings, opts, True,
                                          num_class=opts.num_class, global_step=global_step)
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
                        train_val, loss_val, global_step_val, preds, acc_val = sess.run([
                            train_op, loss_op, global_step, model.preds, model.acc
                        ],
                            feed_dict={model.title_tokens:Y[0], model.scores:Y[2], model.lr:opts.lr})
                        preds_stat = np.concatenate((preds_stat, preds))
                        gt_stat = np.concatenate((gt_stat, Y[2]))
                        loss_stat = np.append(loss_stat, loss_val)
                        if iter_num%1000 == 0:
                            # print('[{}]\tloss:{}\tacc:{:.4f}'.format(iter_num, loss_val, acc_val))
                            for i in range(opts.batch_size):
                                print('====>', vocab.decode(Y[0][i]))
                                print('\tpredict score:\t{}'.format(preds[i]))
                                print('\tground truth score:\t{}'.format(Y[2][i]))
                            acc = np.sum(np.equal(preds_stat, gt_stat)) / preds_stat.size
                            loss = np.sum(loss_stat)/loss_stat.size
                            print('[{}]\tloss:{}\tacc:{:.4f}'.format(iter_num, loss, acc))
                            print('label 1 number is: {}'.format(np.sum(np.equal(preds_stat, 1))))
                            print('label 0 number is: {}'.format(np.sum(np.equal(preds_stat, 0))))
                            preds_stat = np.array([], dtype=np.int32)
                            gt_stat = np.array([], dtype=np.int32)
                            loss_stat = np.array([], dtype=np.float32)
                        iter_num += 1
                        # if acc_val > best_acc:
                        #     best_acc = acc_val
                        #     print('Current best accuracy is: {}'.format(best_acc))
                        #     saver.save(get_session(sess), os.path.join(ckpt_dir, 'best_model'))
                        if iter_num%20000 == 0:
                            best_acc = val(sess, vocab, extractor, model, opts, best_acc, iter_num, ckpt_dir, saver)
                except Exception as e:
                    print(e)
                    saver.save(get_session(sess), os.path.join(ckpt_dir, 'final_model'))

if __name__ == '__main__':
    train()