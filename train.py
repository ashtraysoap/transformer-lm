import os
import sys
import time
import json

import tensorflow as tf

from datagen import TextLoader
from sample import sample_sequence
from model import default_hparams, get_train_ops, Network
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help="path to the directory containing the text corpus")
    parser.add_argument('-m', '--modelpath', type=str, default="models/", help="path under which model checkpoints will be saved")
    parser.add_argument('-l', '--log_dir', type=str, default="logs/", help="path to the dir where logs shall be stored")
    parser.add_argument('-p', '--hparams', type=str, help="path to json-stored hyperparams")
    parser.add_argument('-v', '--verbose', action='store_true', help="if present, prints samples generated while training to stdout")
    parser.add_argument('--stride', type=int, help="offset between succesive dataset instances, defaults to context window size, i.e. no overlap")
    parser.add_argument('--buffer', type=int, help="if specified, reads the input file lazily in chunks of given number of bytes")
    args = parser.parse_args()

    hp = default_hparams()
    if args.hparams is not None:
        with open(args.hparams, 'r') as hf:
            hp.parse_json(hf.read())
    
    batch_size = hp.batch_size

    ds = TextLoader(args.data_dir, seq_length=hp.n_ctx, batch_size=batch_size)
    cti = ds.char_to_idx
    itc = ds.idx_to_char
    hp.n_vocab = len(cti)
    
    network = Network()
    network.construct(hp)
    for i in range(hp.n_epochs):
        print("Epoch ", i)
        network.train_epoch(ds)


if __name__ == "__main__":
    main()