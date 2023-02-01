# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm


class BertEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    here we aditionally add postag embedding.
    """

    def __init__(self, args, bert_embedding):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        # self.max_length = 512
        # self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        # self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        # self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.postag_embedding = nn.Embedding(len(args.postag_vocab), args.emb_size)
        self.bert_embedding = bert_embedding
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, postag_ids, seg, pos=None):
        # word_emb = self.word_embedding(src)
        # if pos is None:
        #     pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, \
        #                                                    dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        # else:
        #     pos_emb = self.position_embedding(pos)
        # seg_emb = self.segment_embedding(seg)
        emb = self.bert_embedding(src, pos, seg)

        postag_emb = self.postag_embedding(postag_ids)

        # emb = word_emb + pos_emb + seg_emb
        emb = emb + postag_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb
