import argparse
import os
import random
import time
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# from uer.utils.tokenizer import BertTokenizer
from brain import KnowledgeGraph
from parser import *
from uer.layers.embeddings import BertEmbedding
# from pretrain import pre_train
# from uer.model_builder import build_model
from uer.model_saver import save_model
from uer.utils.config import load_hyperparam
from uer.utils.constants import *
from uer.utils.seed import set_seed
from uer.utils.vocab import Vocab

warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- FocalLoss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- FocalLoss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# 加入 RNN
class BertClassifier(nn.Module):
    def __init__(self, args):
        super(BertClassifier, self).__init__()
        self.args = args
        pretrain_bert = BertModel.from_pretrained(args.pretrained_model_path)
        # pretrain_bert.encoder.resize_token_embeddings(len(args.vocab.i2w))
        self.embedding = BertEmbedding(args, pretrain_bert.embeddings)
        self.encoder = pretrain_bert.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        rnn = True
        lstm = False
        cnn = False
        add_rnn_or_lstm_or_cnn = rnn or lstm or cnn
        self.add_rnn_or_lstm_or_cnn = add_rnn_or_lstm_or_cnn
        self.rnn = rnn
        self.lstm = lstm
        self.cnn = cnn
        # 加入 RNN或者LSTM
        if add_rnn_or_lstm_or_cnn:
            if rnn:
                self.rnn = nn.RNN(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=2,
                                  batch_first=True, dropout=0.4,
                                  bidirectional=True)
            elif lstm:
                self.rnn = nn.LSTM(input_size=args.hidden_size, hidden_size=args.hidden_size, num_layers=2,
                                   batch_first=True,
                                   dropout=0.4, bidirectional=True)
            if rnn or lstm:
                self.output_layer = nn.Sequential(
                    nn.Linear(args.hidden_size * 2, args.hidden_size // 4),
                    nn.ReLU(),
                    nn.Linear(args.hidden_size // 4, args.labels_num),
                )
            elif cnn:
                self.cnn = nn.Sequential(
                    nn.Conv1d(in_channels=args.hidden_size, out_channels=args.hidden_size, kernel_size=3, stride=1,
                              padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Conv1d(in_channels=args.hidden_size, out_channels=args.hidden_size, kernel_size=3, stride=1,
                              padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                )
                self.output_layer = nn.Sequential(
                    nn.Linear(args.hidden_size, args.hidden_size // 4),
                    nn.ReLU(),
                    nn.Linear(args.hidden_size // 4, args.labels_num),
                )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(args.hidden_size // 4, args.labels_num),
            )

        # 加入预训练
        # if args.pretrain:
        #     self.postag_embedding = nn.Embedding(len(args.postag_vocab), args.emb_size, padding_idx=0)

        # self.output_layer = nn.Linear(args.hidden_size, args.labels_num)

        self.softmax = nn.Softmax(dim=-1)
        self.focalloss = True
        if self.focalloss:
            self.criterion = FocalLoss(alpha=[0.55, 0.45], gamma=2, num_classes=args.labels_num, size_average=False)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.use_vm = False if args.no_vm else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

    def forward(self, src, postag_ids, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """

        if self.args.encoder == "bert":
            if self.args.pretrain:
                # postag_emb = self.postag_embedding(postag_ids)
                emb = self.embedding(src, mask, pos)
                # emb = emb + postag_emb
            else:
                emb = self.embedding(src, postag_ids, mask, pos)
        else:
            emb = self.embedding(src)
        # Encoder.
        if not self.use_vm:
            vm = None
        if self.args.pretrain:
            output = self.encoder(emb).last_hidden_state
        else:
            output = self.encoder(emb, mask, vm)
        # Target.
        # print("output.shape: {}".format(output.shape))
        if not self.cnn:
            if self.pooling == "mean":
                output = torch.mean(output, dim=1)
            elif self.pooling == "max":
                output = torch.max(output, dim=1)[0]
            elif self.pooling == "last":
                output = output[:, -1, :]
            else:
                output = output[:, 0, :]

        # 加入 RNN
        if self.add_rnn_or_lstm_or_cnn:
            if self.rnn or self.lstm:
                output, _ = self.rnn(output.unsqueeze(0))
                output = output.squeeze(0)
            elif self.cnn:
                output = output.permute(0, 2, 1)
                output = self.cnn(output)
                output = output.mean(dim=2)
            # output, _ = self.rnn(output.unsqueeze(0))
            # output = output.squeeze(0)
        # 加入 LSTM
        # output, _ = self.lstm(output.unsqueeze(0))
        # output = output.squeeze(0)
        # print("output.shape: {}".format(output.shape))
        # logits = self.rnn_net(output)
        if not self.focalloss:
            logits = self.softmax(self.output_layer(output))  # [batch_size x labels_num]
            loss = self.criterion(logits.view(-1, self.labels_num), label.view(-1))
            return loss, logits
        else:
            logits = self.output_layer(output)
            loss = self.criterion(logits.view(-1, self.labels_num), label.view(-1))
            return loss, logits


def add_knowledge_worker(params, postag_dict):
    p_id, sentences, columns, kg, vocab, args = params
    sentences_num = len(sentences)
    dataset = []
    for line_id, line in tqdm(enumerate(sentences), desc="Process {}".format(p_id), total=sentences_num):
        # if line_id % 1000 == 0:
        #     print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
        #     sys.stdout.flush()
        # line = line.strip().split('\t')

        label = int(line[1])

        text = CLS_TOKEN + ' ' + line[0]

        tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
        tokens = tokens[0]
        pos = pos[0]
        vm = vm[0].astype("bool")

        token_ids = [vocab.get(t) for t in tokens]

        mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

        # postag 信息
        if args.use_postag:

            postag_ids = [args.postag_vocab.get(t) for t in postag_dict[str(line_id)]]
            # 截断和补齐
            if len(postag_ids) > args.seq_length:
                postag_ids = postag_ids[:args.seq_length]
            else:
                postag_ids = postag_ids + [0] * (args.seq_length - len(postag_ids))

            dataset.append((token_ids, postag_ids, label, mask, pos, vm))
        else:
            dataset.append((token_ids, label, mask, pos, vm))

    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default="bert-base-uncased", type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--dataset_path", default="./data/")
    parser.add_argument("--train_path", type=str, required=False,
                        default="./data/train_data.csv",
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=False, default="./data/dev_data.csv",
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str, required=False, default="./data/test_data.csv",
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    parser.add_argument("--encoder", default="bert", help="Encoder type.")

    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last", "all"], default="all",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["spacy", "bert", "char", "word", "space"], default="spacy",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate. eg: 2e-5")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=1,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=3047,
                        help="Random seed.")

    # Pretrain options
    parser.add_argument("--pretrain_data_path", default="data/pretrain_data.txt")
    parser.add_argument("--checkpoint_save_path", type=str, default="./models/checkpoint")
    parser.add_argument("--pretrain_save_path", type=str, default="./models/bert-base-patent")
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=5)  # 限制checkpoints的数量，最多5个
    parser.add_argument("--pretrain", action="store_true", help="Pretrain the model.")

    # Evaluation options.
    # parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--kg_name", required=False, default="brain/kgs/hate_brains.spo", help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    # extra options
    parser.add_argument("--use_postag", action="store_true", help="Use postag.")
    parser.add_argument("--save_path", type=str, default="./out_models/", help="Path of the output model.")

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    args.output_model_path = f"./models/{args.encoder}_model.bin"
    set_seed(args.seed)

    # Count the number of labels.
    labels_set = dict()
    columns = {}
    # with open(args.train_path, mode="r", encoding="utf-8") as f:
    #     for line_id, line in enumerate(f):
    #         try:
    #             line = line.strip().split("\t")
    #             if line_id == 0:
    #                 for i, column_name in enumerate(line):
    #                     columns[column_name] = i
    #                 continue
    #             label = int(line[columns["label"]])
    #             if label not in labels_set:
    #                 labels_set[label] = 0
    #             labels_set[label] += 1
    #             # labels_set.add(label)
    #         except:
    #             pass
    args.labels_num = 2
    print("labels_num:", args.labels_num)
    for label, count in labels_set.items():
        print("label: %d, count: %d" % (label, count))
    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    # 除了加载原词表外，将数据集和知识图谱中的词也加入词表
    # from uer.utils.tokenizer import BertTokenizer
    # tokenizer = SpacyTokenizer(args)
    # tokenizer = BertTokenizer(args)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    vocab.union(args.train_path, tokenizer, min_count=2)
    vocab.union(args.dev_path, tokenizer, min_count=2)
    vocab.union(args.test_path, tokenizer, min_count=2)
    vocab.union(args.kg_name, tokenizer, min_count=1, type='kg')
    args.vocab = vocab
    # 词性标注词表
    # parse(args)
    postag_vocab = Vocab()
    postag_vocab.load(args.dataset_path + 'postag_vocab.txt')
    args.postag_vocab = postag_vocab
    # 位置词表
    pos_vocab = Vocab()
    pos_vocab.load(args.dataset_path + 'pos_vocab.txt')
    args.pos_vocab = pos_vocab

    # Build bert model.
    # A pseudo target is added.
    # args.target = "bert"
    # model = build_model(args)
    #
    # # 预训练模型
    # model = BertModel.from_pretrained(args.pretrained_model_path)

    # Build classification model.
    model = BertClassifier(args)
    # from models import BertClassifier
    # model = BertClassifier.from_pretrained(args.pretrained_model_path, args=args, num_labels=args.labels_num)

    # Load or initialize parameters.
    # if args.pretrained_model_path is not None and os.path.exists(args.pretrained_model_path):
    #     # Initialize with pretrained model.
    #     model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    # else:
    #     # Initialize with normal distribution.
    #     for n, p in list(model.named_parameters()):
    #         if 'gamma' not in n and 'beta' not in n:
    #             p.data.normal_(0, 0.02)
    # Initialize with normal distribution.
    # for n, p in list(model.named_parameters()):
    #     if 'gamma' not in n and 'beta' not in n:
    #         p.data.normal_(0, 0.02)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    # Datset loader.
    def batch_loader(batch_size, input_ids, postag_ids, label_ids, mask_ids, pos_ids, vms):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i * batch_size: (i + 1) * batch_size, :]
            postag_ids_batch = postag_ids[i * batch_size: (i + 1) * batch_size, :] if postag_ids is not None else None
            label_ids_batch = label_ids[i * batch_size: (i + 1) * batch_size]
            mask_ids_batch = mask_ids[i * batch_size: (i + 1) * batch_size, :]
            pos_ids_batch = pos_ids[i * batch_size: (i + 1) * batch_size, :]
            vms_batch = vms[i * batch_size: (i + 1) * batch_size]
            yield input_ids_batch, postag_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num // batch_size * batch_size:, :]
            postag_ids_batch = postag_ids[instances_num // batch_size * batch_size:,
                               :] if postag_ids is not None else None
            label_ids_batch = label_ids[instances_num // batch_size * batch_size:]
            mask_ids_batch = mask_ids[instances_num // batch_size * batch_size:,
                             :]
            pos_ids_batch = pos_ids[instances_num // batch_size * batch_size:,
                            :]
            vms_batch = vms[instances_num // batch_size * batch_size:]

            yield input_ids_batch, postag_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]
    kg = KnowledgeGraph(args, spo_files=spo_files, predicate=True)

    def read_dataset(path, workers_num=1):

        print("Loading sentences from {}".format(path))
        # sentences = []
        # with open(path, mode='r', encoding="utf-8") as f:
        #     for line_id, line in enumerate(f):
        #         if line_id == 0 or line == "\n":
        #             continue
        #         sentences.append(line)
        # sentence_num = len(sentences)

        # 读取数据集
        dataset = pd.read_csv(path, header=0)
        sentences = dataset.values.tolist()
        sentence_num = len(sentences)

        # 读取语法信息
        with open(path.replace('_data.csv', '_postag.json'), 'r') as f:
            json_dict = json.load(fp=f)

        print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(
            sentence_num, workers_num))
        params = (0, sentences, columns, kg, vocab, args)
        dataset = add_knowledge_worker(params, json_dict)

        # 将数据打乱
        random.shuffle(dataset)

        return dataset

    # Evaluation function.
    def evaluate(args, dataset, is_test, metrics='Acc'):

        metrics = metrics.lower()  # 保证小写
        if args.use_postag:
            input_ids = torch.LongTensor([sample[0] for sample in dataset])
            postag_ids = torch.LongTensor([sample[1] for sample in dataset])
            label_ids = torch.LongTensor([sample[2] for sample in dataset])
            mask_ids = torch.LongTensor([sample[3] for sample in dataset])
            pos_ids = torch.LongTensor([example[4] for example in dataset])
            vms = [example[5] for example in dataset]
            # 把数据截断到最大长度
        else:
            input_ids = torch.LongTensor([sample[0] for sample in dataset])
            postag_ids = None
            label_ids = torch.LongTensor([sample[1] for sample in dataset])
            mask_ids = torch.LongTensor([sample[2] for sample in dataset])
            pos_ids = torch.LongTensor([example[3] for example in dataset])
            vms = [example[4] for example in dataset]

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        if is_test:
            print("The number of test instances: ", instances_num)
        else:
            print("The number of dev instances: ", instances_num)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()
        pbar = tqdm(enumerate(
            batch_loader(batch_size, input_ids, postag_ids, label_ids, mask_ids, pos_ids, vms)),
            total=instances_num)
        for i, (input_ids_batch, postag_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch,
                vms_batch) in pbar:

            # vms_batch = vms_batch.long()
            vms_batch = torch.LongTensor(vms_batch)

            input_ids_batch = input_ids_batch.to(device)
            postag_ids_batch = postag_ids_batch.to(device) if postag_ids_batch is not None else None
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            with torch.no_grad():
                try:
                    _, logits = model(input_ids_batch, postag_ids_batch, label_ids_batch, mask_ids_batch,
                                      pos_ids_batch, vms_batch)
                except Exception as e:
                    print(e)
                    print(input_ids_batch)
                    print(input_ids_batch.size())
                    print(vms_batch)
                    print(vms_batch.size())

            # logits = nn.Softmax(dim=-1)(logits)
            pred = torch.argmax(logits, dim=-1)
            gold = label_ids_batch
            # print(f"pred: {pred}\t gold: {gold}")
            # print(pred)
            for j in range(pred.size()[0]):
                confusion[pred[j], gold[j]] += 1
            correct += torch.sum(pred == gold).item()

            pbar.set_description("Test" if is_test else "Dev")
            pbar.update(1)

        # if is_test:
        #     print("Confusion matrix:")
        #     print(confusion)
        #     print("Report precision, recall, and f1:")
        print("Confusion matrix:")
        print(confusion)
        print("Report precision, recall, and f1:")
        with open(f'./log/{"test_" if is_test else ""}confusion_matrix.txt', 'a') as f:
            f.write("#" * 30)
            f.write('time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            f.write(f"confusion matrix on {'test' if is_test else ''} set: \n")
            f.write(str(confusion) + '\n')

        all_precision = 0
        all_recall = 0
        all_f1 = 0

        for i in range(confusion.size()[0]):
            try:  # 防止除 0 错误
                precision = confusion[i, i].item() / confusion[i, :].sum().item()
            except:
                precision = 0
            try:
                recall = confusion[i, i].item() / confusion[:, i].sum().item()
            except:
                recall = 0
            try:
                f1 = 2 * precision * recall / (precision + recall)
            except:
                f1 = 0
            # if i == 1:
            #     label_1_f1 = f1
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, precision, recall, f1))
            all_precision += precision
            all_recall += recall
            all_f1 += f1

        final_f1 = all_f1 / confusion.size()[0]
        final_precision = all_precision / confusion.size()[0]
        final_recall = all_recall / confusion.size()[0]
        final_acc = correct / instances_num

        print('##########')
        if is_test:
            print("Test result:")
        else:
            print("Dev result:")
        print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
        print("F1. {}".format(final_f1))
        print('precision: {}'.format(final_precision))
        print('Recall: {}'.format(final_recall))
        print('##########')

        # 保存本次实验的结果
        # if not os.path.exists('./log/result.txt'):
        #     os.mknod('./log/result.txt')

        eval_result = {
            'acc': final_acc,
            'f1': final_f1,
            'precision': final_precision,
            'recall': final_recall
        }
        """
        在result.txt中记录每一次实验结果
        在best_result.json中记录最好的实验结果
        """

        # 记录此次实验的结果到result.txt文件中
        with open('./log/result.txt', 'a') as f:
            # 写入实验时间
            # f.write("#" * 30 + "\n")
            f.write("test" + " result:\n" if is_test else "")
            f.write('time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            for key, value in eval_result.items():
                f.write('{}: {}\t'.format(key, value))
            f.write('\n')
            # f.write("#" * 30 + "\n")

        # 以json文件的形式保存每一次的实验结果
        if not os.path.exists('./log/best_result.json'):
            with open('./log/best_result.json', 'w') as f:
                json.dump(eval_result, f)
        else:
            # 保存每一次的实验结果
            with open('./log/best_result.json', 'r') as f:
                old_result = json.load(f)
            # 比较新旧实验结果，如果新实验结果更好，则保存新实验结果
            if eval_result[metrics] > old_result[metrics]:
                # 保存模型checkpoint到args.save_path路径下
                torch.save(model.state_dict(), args.save_path + 'best_model.pt')
                # 保存新实验结果
                with open('./log/best_result.json', 'w') as f:
                    json.dump(eval_result, f)

        if metrics == 'f1':
            return final_f1
        else:
            return final_acc

    # Training phase.
    print("Start training.")
    trainset = read_dataset(args.train_path, workers_num=args.workers_num)
    devset = read_dataset(args.dev_path, workers_num=args.workers_num)
    testset = read_dataset(args.test_path, workers_num=args.workers_num)
    instances_num = len(trainset)
    batch_size = args.batch_size

    print("Trans data to tensor.")
    if args.use_postag:
        input_ids = torch.LongTensor([example[0] for example in trainset])
        postag_ids = torch.LongTensor([example[1] for example in trainset])
        label_ids = torch.LongTensor([example[2] for example in trainset])
        mask_ids = torch.LongTensor([example[3] for example in trainset])
        pos_ids = torch.LongTensor([example[4] for example in trainset])
        vms = [example[5] for example in trainset]
    else:
        input_ids = torch.LongTensor([example[0] for example in trainset])
        label_ids = torch.LongTensor([example[1] for example in trainset])
        mask_ids = torch.LongTensor([example[2] for example in trainset])
        pos_ids = torch.LongTensor([example[3] for example in trainset])
        vms = [example[4] for example in trainset]
        postag_ids = None

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    # if not os.path.exists("./log"):
    #     os.mkdir("./log")
    if not os.path.exists('./log'):
        os.mkdir('./log')

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.01},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = Adam(optimizer_grouped_parameters, lr=args.learning_rate)

    # optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)

    result = 0.0
    best_result = 0.0

    # 构建学习率衰减器
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
    #                                             num_training_steps=train_steps)

    # print(model)
    skip_train = False
    if not skip_train:
        epoch_losses = []
        for epoch in range(1, args.epochs_num + 1):
            model.train()
            total_loss = 0.0
            # 学习率衰减，每个4epoch衰减一次，每次衰减为原来的0.7
            if epoch > 10 and epoch % 4 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5

            pbar = tqdm(enumerate(
                batch_loader(batch_size, input_ids, postag_ids, label_ids, mask_ids, pos_ids, vms)),
                total=int(instances_num / batch_size) + 1)
            for i, (
                    input_ids_batch, postag_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch,
                    vms_batch) in pbar:
                model.zero_grad()

                vms_batch = torch.LongTensor(vms_batch)

                input_ids_batch = input_ids_batch.to(device)
                postag_ids_batch = postag_ids_batch.to(device) if args.use_postag else None
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)

                loss, logits = model(input_ids_batch, postag_ids_batch, label_ids_batch, mask_ids_batch,
                                     pos=pos_ids_batch,
                                     vm=vms_batch)
                # preds = torch.argmax(logits, dim=-1)
                # print(f"preds: {preds}")
                # print(f"label_ids_batch: {label_ids_batch}")
                if torch.cuda.device_count() > 1:
                    loss = torch.mean(loss)
                total_loss += loss.item()
                # if (i + 1) % args.report_steps == 0:
                #     print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                #                                                                       total_loss / args.report_steps))
                #     sys.stdout.flush()
                #     total_loss = 0.

                loss.backward()
                optimizer.step()
                # scheduler.step()
                # scheduler.zero_grad()

                pbar.set_description(
                    "Epoch: {}, Loss: {:.3f} ".format(epoch, loss.item()))
                pbar.update(1)
            epoch_losses.append(total_loss / (i + 1))
            print(f"Epoch {epoch} loss: {total_loss / (i + 1)}")
            print("Start evaluation on dev dataset.")
            result = evaluate(args, devset, False)
            if result > best_result:
                best_result = result
                save_model(model, args.output_model_path)
            else:
                continue
            # only test on test dataset when the model train finished.
            # print("Start evaluation on test dataset.")
            # evaluate(args, True)

        # Evaluation phase.
        print("Final evaluation on the test dataset.")

        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, testset, True)

        # 绘制loss曲线
        plt.plot(epoch_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.savefig('./log/loss_curve.png')
        plt.show()
    else:
        print("Start evaluation on test dataset.")
        evaluate(args, testset, True)


if __name__ == "__main__":
    main()
