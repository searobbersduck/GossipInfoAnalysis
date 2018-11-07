# !/usr/bin/env python3

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='recomend record args')
    parser.add_argument('--max_tokens_per_sent', type=int, default=20)
    parser.add_argument('--resume_max_tokens_per_line', type=int, default=50)
    parser.add_argument('--resume_max_pool_size', type=int, default=300)
    parser.add_argument('--jd_max_pool_size', type=int, default=200)
    parser.add_argument('--jd_max_line', type=int, default=30)
    parser.add_argument('--jd_max_tokens_per_line', type=int, default=50)
    parser.add_argument('--positional_enc_dim', type=int, default=20)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--is_training', type=bool, default=True)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--record_max_length', type=int, default=200)
    parser.add_argument('--num_sampled', type=int, default=400)
    # parser.add_argument('--vocab_size', type=int, required=True)
    parser.add_argument('--vocab_size', type=int, default=8000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--filepattern', default='./file_0_0_*')
    parser.add_argument('--iter_num', type=int, default=10000)
    parser.add_argument('--ckpt_dir', default='./ckpt')
    parser.add_argument('--batch_size', type=int, default=4)
    return parser.parse_args()
