# -*- encoding:utf-8 -*-
import os

import pandas
import torch
from uer.utils.constants import *
from multiprocessing import Pool
from tqdm import tqdm

def count_line(corpus_path):
    count = 0
    with open(corpus_path, mode="r", encoding="utf-8") as f:
        for line in f:
            count += 1
    return count


class Vocab(object):
    """
    """

    def __init__(self):
        self.w2i = {}
        self.i2w = []
        self.w2c = {}
        self.reserved_vocab_path = \
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/reserved_vocab.txt"))

    def load(self, vocab_path, is_quiet=False):
        with open(vocab_path, mode="r", encoding="utf-8") as reader:
            for index, line in enumerate(reader):
                try:
                    w = line.strip().split()[0]
                    self.w2i[w] = index
                    self.i2w.append(w)
                except:
                    self.w2i["???" + str(index)] = index
                    self.i2w.append("???" + str(index))
                    if not is_quiet:
                        print("Vocabulary file line " + str(index + 1) + " has bad format token")
            assert len(self.w2i) == len(self.i2w)
        if not is_quiet:
            print("Vocabulary Size: ", len(self))

    def save(self, save_path):
        print("Vocabulary Size: ", len(self))
        with open(save_path, mode="w", encoding="utf-8") as writer:
            for w in self.i2w:
                writer.write(w + "\n")
        print("Vocabulary saving done.")

    def get(self, w):
        return self.w2i.get(w, UNK_ID)

    def __len__(self):
        return len(self.i2w)

    def worker(self, corpus_path, tokenizer, start, end):
        """ 
        Worker that creates vocabulary from corpus[start:end].
        """
        w2i, i2w, w2c = {}, [], {}
        pos = 0
        with open(corpus_path, mode="r", encoding="utf-8") as f:
            while pos < start:
                try:
                    f.readline()
                except:
                    continue
                finally:
                    pos += 1
            while True:
                try:
                    line = f.readline()
                except:
                    continue
                finally:
                    pos += 1

                tokens = tokenizer.tokenize(line)
                for t in tokens:
                    if t not in w2i:
                        w2i[t], w2c[t] = len(i2w), 1
                        i2w.append(t)
                    else:
                        w2c[t] += 1
                if pos >= end - 1:
                    return (w2i, i2w, w2c)


    def union(self, corpus_path, tokenizer, min_count=3, type='dataset'):
        """将数据集分词词频 >3 的词后与原词表合并"""
        print("vocab size: {} before union".format(len(self.i2w)))
        frequency = {}
        if type == 'dataset':
            df = pandas.read_csv(corpus_path, header=0, encoding='utf-8')
            tweets = df['tweet'].values.tolist()
            for tweet in tqdm(tweets, desc="union vocab", total=len(tweets)):
                tokens = tokenizer.tokenize(tweet)
                for t in tokens:
                    if t not in frequency:
                        frequency[t] = 1
                    else:
                        frequency[t] += 1
        else:
            with open(corpus_path, 'r', encoding="utf-8") as f:
                for line_id, line in tqdm(enumerate(f.readlines()), desc="union", total=len(f.readlines())):
                    a, b, c = line.split('\t')
                    text = a + ' ' + b + ' ' + c
                    text = tokenizer.tokenize(text)
                    for t in text:
                        if t in frequency.keys():
                            frequency[t] += 1
                        else:
                            frequency[t] = 1

        vocab_list = [k for k, v in frequency.items() if v >= min_count]
        for word in vocab_list:
            word = word.lower()
            if word not in self.i2w:
                self.i2w.append(word)
            if word not in self.w2i.keys():
                self.w2i[word] = len(self.w2i)
        print("vocab size: {} after union".format(len(self.i2w)))


def build(self, corpus_path, tokenizer, workers_num=1, min_count=1):
    """ Build vocabulary from the given corpus. """
    print("Start %d workers for building vocabulary..." % workers_num)
    lines_num = count_line(corpus_path)
    pool = Pool(workers_num)
    vocab_list = []
    for i in range(workers_num):
        start = i * lines_num // workers_num
        end = (i + 1) * lines_num // workers_num
        vocab_p = pool.apply_async(func=self.worker, args=[corpus_path, tokenizer, start, end])
        vocab_list.append(vocab_p.get())
    pool.close()
    pool.join()

    # Union vocab in all workers.
    w2i, i2w, w2c = self.union(vocab_list)
    # Sort w2c according to word count.
    sorted_w2c = sorted(w2c.items(), key=lambda item: item[1], reverse=True)

    # Add special symbols and remove low frequency words.
    with open(self.reserved_vocab_path, mode="r", encoding="utf-8") as reader:
        self.i2w = [line.strip().split()[0] for line in reader]

    for i, w in enumerate(self.i2w):
        self.w2i[w] = i
        self.w2c[w] = -1

    for w, c in sorted_w2c:
        if c < min_count:
            break
        if w not in self.w2i:
            self.w2i[w], self.w2c[w] = len(self.i2w), c
            self.i2w.append(w)
