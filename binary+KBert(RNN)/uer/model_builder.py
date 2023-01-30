# -*- encoding:utf-8 -*-
import torch
from uer.layers.embeddings import BertEmbedding, Embedding
from uer.encoders.bert_encoder import BertEncoder
from uer.targets.bert_target import BertTarget
from uer.models.model import Model
from pretrain import pre_train


def build_model(args):
    """
    Build universial encoder representations models.
    The combinations of different embedding, encoder, 
    and target layers yield pretrained models of different 
    properties. 
    We could select suitable one for downstream tasks.
    """

    if args.subword_type != "none":
        subencoder = globals()[args.subencoder.capitalize() + "Subencoder"](args, len(args.sub_vocab))
    else:
        subencoder = None
    embedding = BertEmbedding(args, len(args.vocab)) if args.encoder == "bert" else Embedding(args, len(args.vocab))
    encoder = globals()[args.encoder.capitalize() + "Encoder"](args)
    target = globals()[args.target.capitalize() + "Target"](args, len(args.vocab))  # 分类器中并没有使用
    model = Model(args, embedding, encoder, target, subencoder)

    return model
