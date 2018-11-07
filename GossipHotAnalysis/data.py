# !/usr/bin/env python3

import os
import numpy as np
import json
from glob import glob
import random

import sys
sys.path.append('./')
sys.path.append('./corpus/w2v')

import gen_pieces
import vocab as vocab_160k

# 1. convert json to txt
def convert_gossip_json_to_txt(infile, outdir):
    with open(infile, 'r', encoding='utf8') as f:
        titles_list = []
        comment_nums_list = []
        for line in f.readlines():
            line = line.strip()
            if line is None or line == '':
                continue
            title_json = json.loads(line)
            if 'data' not in title_json:
                continue
            title = title_json['data']['title']
            comment_num = title_json['data']['comment_num']
            titles_list.append(title)
            comment_nums_list.append(str(comment_num))
    basename = os.path.basename(infile).split('.')[:-1]
    basename = '_'.join(basename)
    out_title_file = os.path.join(outdir, basename+'_title.txt')
    out_comment_file = os.path.join(outdir, basename+'_comment_num.txt')
    os.makedirs(outdir, exist_ok=True)
    assert len(titles_list) == len(comment_nums_list)
    with open(out_title_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(titles_list))
    print('====> write title file:\t{}'.format(out_title_file))
    with open(out_comment_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(comment_nums_list))
    print('====> write comment number file:\t{}'.format(out_comment_file))

def test_convert_gossip_json_to_txt():
    infiles = glob('./dataset/comment/*')
    outdir = os.path.join('./dataset', 'comment_titles')
    for infile in infiles:
        convert_gossip_json_to_txt(infile, outdir)

# 2. seg txt

'''
spm_encode --model=.GossipHotAnalysis/corpus/w2v/big_160k_spm.model --output_format=piece < ./file_0_0.txt > ./file_0_0_seg.txt
'''
def generate_seg_script():
    infilepattern = './dataset/comment_titles/comment_part*_title.txt'
    model_dir = './corpus/w2v/big_160k_spm.model'
    out_seg_dir = './dataset/comment_titles_seg'
    os.makedirs(out_seg_dir,exist_ok=True)
    out_script_file = './generate_seg_script.sh'
    gen_pieces.generate_seg_script(infilepattern, model_dir, out_seg_dir, out_script_file)

# 3. merge segemented title and its comment number into one item
def merge_seg_titles_and_comment_num(in_title_file, in_comment_num_file):
    len1 = len(open(in_title_file, 'r', encoding='utf8').readlines())
    len2 = len(open(in_comment_num_file, 'r', encoding='utf8').readlines())
    assert len1 == len2
    titles_list = []
    comment_num_list = []
    with open(in_title_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or line == '':
                continue
            titles_list.append(line)
    with open(in_comment_num_file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or line == '':
                continue
            comment_num_list.append(line)
    titles_comments_list = zip(titles_list, comment_num_list)
    result_list = []
    comment_num_max = -1
    comment_num_min = 1000000
    for elem in titles_comments_list:
        dict = {}
        dict['title'] = elem[0]
        dict['comment_num'] = elem[1]
        if int(elem[1]) > comment_num_max:
            comment_num_max = int(elem[1])
        if int(elem[1]) < comment_num_min:
            comment_num_min = int(elem[1])
        result_list.append(dict)
    return result_list, comment_num_min, comment_num_max

def test_merge_seg_titles_and_comment_num():
    title_filepattern = './dataset/comment_titles_seg/comment_part*_title.txt'
    comment_nums_filepattern = './dataset/comment_titles/comment_part*_comment_num.txt'
    outdir = './dataset/use_data'
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, 'gossip_data.txt')
    titles = glob(title_filepattern)
    comment_nums = glob(comment_nums_filepattern)
    result_list = []
    num_max = -1
    num_min = 1000000
    for i in range(len(titles)):
        res, num_min1, num_max1 = merge_seg_titles_and_comment_num(
            comment_nums[i].replace('comment_titles', 'comment_titles_seg').replace('_comment_num', '_title'), comment_nums[i])
        result_list += res
        num_max = max(num_max, num_max1)
        num_min = min(num_min, num_min1)
    str_list = []
    for res in result_list:
        res['score'] = (int(res['comment_num']) - num_min) / (num_max - num_min)
        str_list.append(json.dumps(res, ensure_ascii=False))
    with open(outfile, 'w', encoding='utf8') as f:
        f.write('\n'.join(str_list))
    print('hello world!')

# 4. load data
class GossipCommentNumberExtractor(object):
    def __init__(self):
        self.name = 'GossipCommentNumberExtractor'

    def extract(self, infile):
        articles = []
        with open(infile, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or line == '':
                    continue
                articles.append(json.loads(line))
        return articles

def _get_batch(generator, vocab, batch_size, tokens_per_line):
    while True:
        titles_inputs = np.zeros(
            [batch_size, tokens_per_line], dtype=np.int32
        )
        comment_nums_inputs = np.zeros(
            [batch_size], dtype=np.float32
        )
        for i in range(batch_size):
            article = next(generator)
            title_tokens = vocab.encode(article['title'])
            cut_len = min(tokens_per_line, len(title_tokens))
            titles_inputs[i, :cut_len] = title_tokens[:cut_len]
            comment_nums_inputs[i] = article['score']
        yield titles_inputs, comment_nums_inputs

class GossipCommentNumberDataset(object):
    def __init__(self, filepattern,
                 vocab,
                 extractor,
                 test=False,
                 shuffle_on_load=False):
        self._vocab = vocab
        self._extractor = extractor
        self._all_shards = glob(filepattern)
        print('Found {} shards at {}'.format(len(self._all_shards), filepattern))
        self._shards_to_choose = []
        self._test = test
        self._shuffle_on_load = shuffle_on_load
        self._infos = self._load_random_shard()

    def _choose_random_shard(self):
        if len(self._shards_to_choose) == 0:
            self._shards_to_choose = list(self._all_shards)
            random.shuffle(self._shards_to_choose)
        shard_name = self._shards_to_choose.pop()
        return shard_name

    def _load_shard(self, shard_name):
        print('load data from {}'.format(shard_name))
        articles = self._extractor.extract(shard_name)
        if self._shuffle_on_load:
            ids = [i for i in range(len(articles))]
            random.shuffle(ids)
            articles = [articles[i] for i in ids]
        return articles

    def _load_random_shard(self):
        if self._test:
            if len(self._all_shards) == 0:
                raise StopIteration
            else:
                shard_name = self._all_shards.pop()
        else:
            shard_name = self._choose_random_shard()
        infos = self._load_shard(shard_name)
        self._i = 0
        self._ninfos = len(infos)
        return infos

    def get_sentence(self):
        while True:
            if self._i == self._ninfos:
                self._infos = self._load_random_shard()
            ret = self._infos[self._i]
            self._i += 1
            yield ret

    def iter_batches(self, batch_size, tokens_per_line):
        for X in _get_batch(self.get_sentence(), self.vocab, batch_size, tokens_per_line):
            yield X

    @property
    def vocab(self):
        return self._vocab

def test_GossipCommentNumberDataset():
    extractor = GossipCommentNumberExtractor()
    vocab = vocab_160k.Vocabulary('./corpus/w2v/v160k_big_string.txt')
    ds = GossipCommentNumberDataset('./dataset/use_data/*.txt', vocab, extractor)
    tokens_per_sent = 20
    for i in range(1000):
        X = ds.iter_batches(1, tokens_per_sent)
        Y = next(X)
        print(Y[0])
        print('hello world!')


if __name__ == '__main__':
    # test_convert_gossip_json_to_txt()
    # generate_seg_script()
    test_merge_seg_titles_and_comment_num()
    test_GossipCommentNumberDataset()