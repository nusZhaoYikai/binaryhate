import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score
from torch import optim
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

data_dir = "./data"
train_file = os.path.join(data_dir, "train_data.csv")
dev_file = os.path.join(data_dir, "dev_data.csv")
test_file = os.path.join(data_dir, "test_data.csv")


class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.file_path = file_path
        self.data = self.read_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def read_data(self):
        data = pd.read_csv(self.file_path, header=0)
        data = data.values.tolist()
        output = []
        for text, label in data:
            text = self.tokenizer.encode_plus(text, max_length=self.max_len, padding='max_length', truncation=True)
            output.append((torch.tensor(text['input_ids']), torch.tensor(text['attention_mask']), torch.tensor(label)))
        return output


def get_dataloader(batch_size=768, max_len=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print(f"vocab_size: {tokenizer.vocab_size}")
    train_dataset = MyDataset(train_file, tokenizer, max_len=max_len)
    dev_dataset = MyDataset(dev_file, tokenizer, max_len=max_len)
    test_dataset = MyDataset(test_file, tokenizer, max_len=max_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader, test_loader


#
#
class SVMmodel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, input_dim, output_dim):
        super(SVMmodel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.criteria = nn.MultiMarginLoss()

    def forward(self, x, label=None):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x.float())
        logit = F.softmax(x, -1)

        if label is not None:
            loss = self.criteria(logit.view(-1, 2), label.view(-1))
            return loss, logit
        else:
            return None, logit


# class SVMmodel():
#     def __init__(self):
#         super(SVMmodel, self).__init__()
#         self.fc = SVC(kernel='linear')
#         self.criteria = nn.MultiMarginLoss()
#
#     def fit(self, x, y):
#         self.fc.fit(x, y)
#
#     def predict(self, x):
#         return self.fc.predict(x)
#
#     def score(self, x, y):
#         return self.fc.score(x, y)

def train(args, model, train_loader, dev_loader, optimizer, device, epoch, save_path="./out_models"):
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    print("start train epoch: {}".format(epoch), flush=True)
    for i, batch in pbar:
        # print(batch)
        input_ids, attention_mask, label = batch
        input_ids = input_ids.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        loss, logits = model(input_ids, label=label)
        if torch.cuda.device_count() > 1:
            loss = torch.mean(loss)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"train loss: {loss.item():.4f}")
        pbar.update(1)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        pbar = tqdm(enumerate(dev_loader), total=len(dev_loader))
        all_labels = []
        all_preds = []
        confusion = torch.zeros(2, 2, dtype=torch.long)
        for i, batch in pbar:
            input_ids, attention_mask, label = batch
            input_ids = input_ids.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            _, logits = model(input_ids, label=label)
            pred = torch.argmax(logits, dim=1)
            all_labels.extend(label.cpu().numpy().tolist())
            all_preds.extend(pred.cpu().numpy().tolist())
            correct += torch.sum(pred == label).item()
            total += len(label)
            f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro')
            presion = precision_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro')
            recall = recall_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro')
            pbar.set_description("dev acc: {} f1: {} presion: {} recall: {}".format(correct / total, f1,
                                                                                    presion,
                                                                                    recall))
            pbar.update(1)
        for j in range(len(all_labels)):
            confusion[all_preds[j], all_labels[j]] += 1

        print("Confusion matrix:")
        print(confusion)
        with open(f'./log/confusion_matrix.txt', 'a') as f:
            f.write("#" * 30)
            f.write('time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            f.write(f"confusion matrix on dev set: \n")
            f.write(str(confusion) + '\n')

        final_acc = correct / total
        final_f1 = f1_score(all_labels, all_preds, average='macro')
        final_presion = precision_score(all_labels, all_preds, average='macro')
        final_recall = recall_score(all_labels, all_preds, average='macro')
        eval_result = {
            'acc': final_acc,
            'f1': final_f1,
            'precision': final_presion,
            'recall': final_recall
        }
        # print(eval_result)
        # ??????????????????????????????result.txt?????????
        with open('./log/result.txt', 'a') as f:
            # ??????????????????
            # f.write("#" * 30 + "\n")
            # f.write(" result:")
            f.write('time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            for key, value in eval_result.items():
                f.write('{}: {}\t'.format(key, value))
            f.write('\n')

        print(f"epoch {epoch} classification report: \n {classification_report(all_labels, all_preds)}")
        """
        ???best_result.json??????????????????????????????
        """

        # ???json?????????????????????????????????????????????
        if not os.path.exists(f'./log/best_{args.model_name}_result.json'):
            with open(f'./log/best_{args.model_name}_result.json', 'w') as f:
                json.dump(eval_result, f)
        else:
            # ??????????????????????????????
            with open(f'./log/best_{args.model_name}_result.json', 'r') as f:
                old_result = json.load(f)
            # ?????????????????????????????????????????????????????????????????????????????????
            if eval_result["f1"] > old_result["f1"]:
                # ????????????checkpoint???args.save_path?????????
                torch.save(model.state_dict(), save_path + f'best_{args.model_name}_model.pt')
                # ?????????????????????
                with open(f'./log/best_{args.model_name}_result.json', 'w') as f:
                    json.dump(eval_result, f)


def predict(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        all_preds = []
        pbar = tqdm(test_loader)
        all_labels = []
        confusion = torch.zeros(2, 2, dtype=torch.long)
        for i, (input_ids, attention_mask, label) in enumerate(pbar):
            input_ids = input_ids.to(device)
            # attention_mask = attention_mask.to(device)
            # label = label.to(device)
            _, logits = model(input_ids)
            pred = torch.argmax(logits, dim=1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(label.cpu().numpy().tolist())
    for j in range(len(all_labels)):
        confusion[all_preds[j], all_labels[j]] += 1

    print("Confusion matrix:")
    print(confusion)
    with open(f'./log/test_confusion_matrix.txt', 'a') as f:
        f.write("#" * 30)
        f.write('time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.write(f"confusion matrix on test set: \n")
        f.write(str(confusion) + '\n')
    # ??????acc\presion\recall\f1
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    eval_result = {
        'acc': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    # ??????????????????????????????result.txt?????????
    with open('./log/result.txt', 'a') as f:
        # ??????????????????
        # f.write("#" * 30 + "\n")
        f.write("test result:\n")
        f.write('time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for key, value in eval_result.items():
            f.write('{}: {}\t'.format(key, value))
        f.write('\n')

        # ???json?????????????????????????????????????????????
        if not os.path.exists('./log/best_result.json'):
            with open('./log/best_result.json', 'w') as f:
                json.dump(eval_result, f)
        else:
            # ??????????????????????????????
            with open('./log/best_result.json', 'r') as f:
                old_result = json.load(f)
            # ?????????????????????????????????????????????????????????????????????????????????
            if eval_result['f1'] > old_result['f1']:
                # ????????????checkpoint???args.save_path?????????
                torch.save(model.state_dict(), './out_models/' + 'best_model.pt')
                # ?????????????????????
                with open('./log/best_result.json', 'w') as f:
                    json.dump(eval_result, f)
    # print("final test acc: {} f1: {} presion: {} recall: {}".format(acc, f1, presion, recall))
    print(f"final test classification report: \n {classification_report(all_labels, all_preds)}")


# def train(args, model, train_loader, dev_loader, optimizer, device, epoch, save_path="./out_models"):
#     model.train()
#     pbar = tqdm(enumerate(train_loader), total=len(train_loader))
#     print("start train epoch: {}".format(epoch))
#     for i, (input_ids, attention_mask, label) in pbar:
#         input_ids = input_ids.to(device)
#         label = label.to(device)
#         optimizer.zero_grad()
#         loss, logits = model(input_ids, label=label)
#         loss.backward()
#         optimizer.step()
#         pbar.set_description(f"train loss: {loss.item():.4f}")
#         pbar.update(1)
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         pbar = tqdm(dev_loader)
#         all_labels = []
#         all_preds = []
#         for i, (input_ids, attention_mask, label) in enumerate(pbar):
#             input_ids = input_ids.to(device)
#             label = label.to(device)
#             _, logits = model(input_ids)
#             pred = torch.argmax(logits, dim=1)
#             all_labels.extend(label.cpu().numpy().tolist())
#             all_preds.extend(pred.cpu().numpy().tolist())
#             correct += torch.sum(pred == label).item()
#             total += len(label)
#             f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro', zero_division=0)
#             presion = precision_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro', zero_division=0)
#             recall = recall_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro', zero_division=0)
#             pbar.set_description("dev acc: {} f1: {} presion: {} recall: {}".format(correct / total, f1,
#                                                                                     presion,
#                                                                                     recall))
#             pbar.update(1)
#         final_acc = correct / total
#         final_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
#         final_presion = precision_score(all_labels, all_preds, average='macro', zero_division=0)
#         final_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
#         eval_result = {
#             'acc': final_acc,
#             'f1': final_f1,
#             'precision': final_presion,
#             'recall': final_recall
#         }
#         # print(eval_result)
#
#         print(
#             f"epoch {epoch} classification report: \n {classification_report(all_labels, all_preds, zero_division=0)}")
#         """
#         ???best_result.json??????????????????????????????
#         """
#
#     # ???json?????????????????????????????????????????????
#     if not os.path.exists(f'./log/best_{args.model_name}_result.json'):
#         with open(f'./log/best_{args.model_name}_result.json', 'w') as f:
#             json.dump(eval_result, f)
#     else:
#         # ??????????????????????????????
#         with open(f'./log/best_{args.model_name}_result.json', 'r') as f:
#             old_result = json.load(f)
#         # ?????????????????????????????????????????????????????????????????????????????????
#         if eval_result["f1"] > old_result["f1"]:
#             # ????????????checkpoint???args.save_path?????????
#             torch.save(model.state_dict(), save_path + f'best_{args.model_name}_model.pt')
#             # ?????????????????????
#             with open(f'./log/best_{args.model_name}_result.json', 'w') as f:
#                 json.dump(eval_result, f)


# def predict(model, test_loader, device):
#     # model.eval()
#
#     # all_input_ids = []
#     # all_labels = []
#     #
#     # for i, (input_ids, att, label) in enumerate(test_loader):
#     #     all_input_ids.extend(input_ids.numpy().tolist())
#     #     all_labels.extend(label.numpy().tolist())
#     # all_input_ids = np.array(all_input_ids)
#     # all_labels = np.array(all_labels)
#     # preds = model.predict(all_input_ids)
#     # print(f"test score: {model.score(all_input_ids, all_labels)}")
#     # print(f"test classification report: \n {classification_report(all_labels, preds)}")
#     model.eval()
#     with torch.no_grad():
#         preds = []
#         pbar = tqdm(test_loader)
#         labels = []
#         for i, (input_ids, attention_mask, label) in enumerate(pbar):
#             input_ids = input_ids.to(device)
#             attention_mask = attention_mask.to(device)
#             _, logits = model(input_ids, attention_mask)
#             pred = torch.argmax(logits, dim=1)
#             preds.extend(pred.cpu().numpy().tolist())
#             labels.extend(label.cpu().numpy().tolist())
#         # ??????acc\presion\recall\f1
#         # acc = accuracy_score(labels, preds)
#         # f1 = f1_score(labels, preds, average='macro')
#         # presion = precision_score(labels, preds, average='macro')
#         # recall = recall_score(labels, preds, average='macro')
#         # print("final test acc: {} f1: {} presion: {} recall: {}".format(acc, f1, presion, recall))
#         print(f"final test classification report: \n {classification_report(labels, preds, zero_division=0)}")


def main():
    passer = argparse.ArgumentParser()
    passer.add_argument('--batch_size', type=int, default=768)
    passer.add_argument('--lr', type=float, default=2e-4)
    passer.add_argument('--epochs', type=int, default=20)
    passer.add_argument('--save_path', type=str, default='./out_models/')
    passer.add_argument('--model_name', choices=["baseline_bert", "cnn", "lstm", "svm"], default="svm", type=str,
                        help="?????????")
    passer.add_argument('--seq_len', type=int, default=64, help="????????????")

    args = passer.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists('./log'):
        os.mkdir('./log')

    # ??????????????????
    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)
    # ????????????
    print("??????????????????")
    t1 = time.time()
    train_loader, dev_loader, test_loader = get_dataloader(args.batch_size, args.seq_len)
    print(f"??????????????????, ??????{time.time() - t1}s")
    # ????????????,vocab_size??????BertTokenizer???vocab_size
    print("??????????????????")
    vocab_size = 30522
    model = SVMmodel(vocab_size, 64, 64 * args.seq_len, 2)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    # ??????GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # ???????????????
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    # ????????????
    print("??????????????????")
    t2 = time.time()
    for epoch in range(args.epochs):
        train(args, model, train_loader, dev_loader, optimizer, device, epoch)
    print(f"??????????????????, ??????{time.time() - t2}s")
    # ??????
    save_path = './out_models/'
    if os.path.exists(save_path + f'best_{args.model_name}_model.pt'):
        model.load_state_dict(torch.load(save_path + f'best_{args.model_name}_model.pt'))
    print("????????????")
    predict(model, test_loader, device)
    print(f"????????????,?????????{time.time() - t1}s")


if __name__ == '__main__':
    main()
