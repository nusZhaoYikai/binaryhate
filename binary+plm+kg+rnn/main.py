#!/usr/bin/env python
# coding=utf-8
import argparse
import json
import time

import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from tqdm import tqdm

from data_utils import load_data_agument
from models import *
from utils import *

warnings.filterwarnings("ignore")


def init_args():
    parser = argparse.ArgumentParser()

    # data options
    DATA_PATHS = {'plm+kg-hate-binary': "data/plm+kg-hate-corpus-v1/binary/",
                  'plm+kg-hate-stg1': "data/plm+kg-hate-corpus-v1/stg1/",
                  'plm+kg-hate-stg2': "data/plm+kg-hate-corpus-v1/stg2/",
                  "plm+kg": "data/",
                  }
    parser.add_argument("--dataset", default="plm+kg", type=str,
                        help="The dataset to run")
    parser.add_argument("--seq_len", default=128, type=int,
                        help="Max Sequence Length")
    parser.add_argument("--use_kg", default=True, type=bool,
                        help="Whether to use knowledge graph.")
    parser.add_argument("--kg_path", default="data/kgs/hate_brains.spo", type=str,
                        help="knowledge graph path.")
    parser.add_argument("--data_balance", default=True, type=bool,
                        help="Whether to balance data according to labels.")

    # train options
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size. eg: 16, 32, 64")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate. eg: 2e-5, 1e-5, 1e-4")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed.")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')

    # model options
    parser.add_argument('--model', type=str, default="bert-lstm", choices=["bert-lstm"])
    parser.add_argument("--plm_type", type=str, default="bert", help="The pretrained model type.")
    parser.add_argument("--plm", default="bert", type=str, choices=["bert", "hateBERT"],
                        help="The pretrained model name.")
    parser.add_argument("--savemodel_path", default="models/best_model.pt", type=str)

    parser.add_argument("--resume_training", action="store_true", help="Whether to resume training from a checkpoint.")
    parser.add_argument("--bidirectional", action="store_true", help="Whether to use bidirectional LSTM.")

    MODEL_PATHS = {"bert": "bert-base-uncased",
                   "hateBERT": "./models/hateBERT"}
    # prompt options
    # parser.add_argument("--shot", default="16", type=str, help="X-shot for prompt learning",
    #                     choices=["0", "16", "64", "256", "1024", "all"])

    args = parser.parse_args()

    args.plm_path = MODEL_PATHS[args.plm]
    args.data_path = DATA_PATHS[args.dataset]

    return args


def train(args, train_data, dev_data):
    model = HateClassifier(args).to(args.device)
    model.train()

    if args.resume_training and os.path.exists(args.savemodel_path):
        model.load_state_dict(torch.load(args.savemodel_path))
        print("Resume training from {}".format(args.savemodel_path))

    # weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    for i in range(args.epochs):
        print('[Training] Epoch:{}'.format(i))
        total_loss = 0
        for batch_id, batch in enumerate(tqdm(train_data)):
            input_ids, mask_ids, true_labels = \
                batch["input_ids"].to(args.device), batch["masks"].to(args.device), batch['labels'].to(args.device)

            pred_labels = model(input_ids, mask_ids)

            loss = criterion(pred_labels, true_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # print(loss.item())

        print("Epoch {}, total loss: {:.4f}\n".format(i, total_loss))
        p, r, f1 = eval(args, eval_data=dev_data, model=model)

        # 保存在验证集集上最优的 epoch
        # if f1 > best_f1:
        #     print("Best model saving in {}...".format(args.savemodel_path))
        #     torch.save(model, args.savemodel_path)
        #     best_f1 = f1
        best_f1 = max(best_f1, f1)


def eval(args, eval_data, model=None, type='dev', metrics='acc'):
    if model is None:
        model = torch.load(args.savemodel_path).to(args.device)
    model.eval()
    # Confusion matrix.
    confusion = torch.zeros(2, 2, dtype=torch.long)

    all_true_labels, all_pred_labels = [], []
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(eval_data)):
            input_ids, mask_ids = batch["input_ids"].to(args.device), batch["masks"].to(args.device)
            true_labels = batch['labels']

            pred_labels = model(input_ids, mask_ids)
            preds = pred_labels.argmax(dim=-1).detach()
            for j in range(pred_labels.size()[0]):
                confusion[preds[j], true_labels[j]] += 1

            all_true_labels += true_labels.detach().numpy().tolist()
            all_pred_labels += pred_labels.argmax(dim=-1).detach().cpu().numpy().tolist()

    precision = precision_score(all_true_labels, all_pred_labels, average="macro")
    recall = recall_score(all_true_labels, all_pred_labels, average="macro")
    f1 = f1_score(all_true_labels, all_pred_labels, average="macro")
    acc = accuracy_score(all_true_labels, all_pred_labels)

    eval_result = {"precision": precision, "recall": recall, "f1": f1, "acc": acc}
    print(f"classification report on {type} set:")
    print(classification_report(all_true_labels, all_pred_labels, digits=4))
    print(f"confusion matrix on {type} set:")
    print(confusion)

    # 将混淆矩阵打印到文件中
    if not os.path.exists("./log"):
        os.mkdir("./log")
    with open('./log/confusion_matrix.txt', 'a') as f:
        f.write("#" * 30)
        f.write('time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.write(f"confusion matrix on {type} set: \n")
        f.write(str(confusion)+'\n')

    """
    在result.text中保存每一次的结果
    在best_result.json中保存最好的结果
    """
    # 记录此次实验的结果到result.txt文件中
    # if not os.path.exists("./log"):
    #     os.mkdir("./log")
    with open('./log/result.txt', 'a') as f:
        # 写入实验时间
        f.write("#" * 30)
        f.write('time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for key, value in eval_result.items():
            f.write('{}: {}\t'.format(key, value))
        f.write("#" * 30 + "\n")

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
            torch.save(model, args.savemodel_path)
            # 保存新实验结果
            with open('./log/best_result.json', 'w') as f:
                json.dump(eval_result, f)

    if type == 'dev':
        print("[Evaluation] on dev dataset:\n P: {:.4f}, R: {:.4f}, F1: {:.4f}".format(precision, recall, f1))
    elif type == 'test':
        print(
            "\n\n[Final evaluation] on test dataset:\n P: {:.4f}, R: {:.4f}, F1: {:.4f}".format(precision, recall, f1))

    return precision, recall, f1


if __name__ == '__main__':
    args = init_args()
    print(args)
    if args.seed is not None:
        set_seed(args.seed)

    if not os.path.exists("./log"):
        os.mkdir("./log")
    else:
        with open('./log/result.txt', 'a') as f:
            f.write("\n\n")

    train_dataloader, test_dataloader, dev_dataloader = load_data_agument(args)

    if args.mode == 'train':
        train(args, train_dataloader, dev_dataloader)
        eval(args, test_dataloader, type="test")
    else:
        eval(args, test_dataloader, type="test")
