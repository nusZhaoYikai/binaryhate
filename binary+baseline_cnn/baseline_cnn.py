"""
使用bert进行文本分类
"""
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
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from torch.optim import AdamW
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


def get_dataloader(batch_size=768):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(f"vocab_size: {tokenizer.vocab_size}")
    train_dataset = MyDataset(train_file, tokenizer)
    dev_dataset = MyDataset(dev_file, tokenizer)
    test_dataset = MyDataset(test_file, tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader, test_loader


class CNNModel(nn.Module):
    def __init__(self, vocab_size, static=True):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 768)
        kernel_sizes = [2, 3, 4]
        kernel_num = 100
        # kernel_num等于bert的tokenizer的vocab_size
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, kernel_num, (k, 768)) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, 2)
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x, att, label=None):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = F.softmax(self.fc(x), -1)

        if label is not None:
            loss = self.criteria(logit.view(-1, 2), label.view(-1))
            return loss, logit
        else:
            return None, logit


def train(args, model, train_loader, dev_loader, optimizer, device, epoch, save_path="./out_models"):
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    print("start train epoch: {}".format(epoch))
    for i, (input_ids, attention_mask, label) in pbar:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        loss, logits = model(input_ids, attention_mask, label=label)
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
        pbar = tqdm(dev_loader)
        all_labels = []
        all_preds = []
        confusion = torch.zeros(2, 2, dtype=torch.long)
        for i, (input_ids, attention_mask, label) in enumerate(pbar):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)
            _, logits = model(input_ids, attention_mask)
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
        # 记录此次实验的结果到result.txt文件中
        with open('./log/result.txt', 'a') as f:
            # 写入实验时间
            # f.write("#" * 30 + "\n")
            # f.write(" result:")
            f.write('time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            for key, value in eval_result.items():
                f.write('{}: {}\t'.format(key, value))
            f.write('\n')

        print(f"epoch {epoch} classification report: \n {classification_report(all_labels, all_preds)}")
        """
        在best_result.json中记录最好的实验结果
        """

        # 以json文件的形式保存每一次的实验结果
        if not os.path.exists(f'./log/best_{args.model_name}_result.json'):
            with open(f'./log/best_{args.model_name}_result.json', 'w') as f:
                json.dump(eval_result, f)
        else:
            # 保存每一次的实验结果
            with open(f'./log/best_{args.model_name}_result.json', 'r') as f:
                old_result = json.load(f)
            # 比较新旧实验结果，如果新实验结果更好，则保存新实验结果
            if eval_result["f1"] > old_result["f1"]:
                # 保存模型checkpoint到args.save_path路径下
                torch.save(model.state_dict(), save_path + f'best_{args.model_name}_model.pt')
                # 保存新实验结果
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
            attention_mask = attention_mask.to(device)
            # label = label.to(device)
            _, logits = model(input_ids, attention_mask)
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
    # 计算acc\presion\recall\f1
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
    # 记录此次实验的结果到result.txt文件中
    with open('./log/result.txt', 'a') as f:
        # 写入实验时间
        # f.write("#" * 30 + "\n")
        f.write("test result:\n")
        f.write('time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for key, value in eval_result.items():
            f.write('{}: {}\t'.format(key, value))
        f.write('\n')

        # 以json文件的形式保存每一次的实验结果
        if not os.path.exists('./log/best_result.json'):
            with open('./log/best_result.json', 'w') as f:
                json.dump(eval_result, f)
        else:
            # 保存每一次的实验结果
            with open('./log/best_result.json', 'r') as f:
                old_result = json.load(f)
            # 比较新旧实验结果，如果新实验结果更好，则保存新实验结果
            if eval_result['f1'] > old_result['f1']:
                # 保存模型checkpoint到args.save_path路径下
                torch.save(model.state_dict(), './out_models/' + 'best_model.pt')
                # 保存新实验结果
                with open('./log/best_result.json', 'w') as f:
                    json.dump(eval_result, f)
    # print("final test acc: {} f1: {} presion: {} recall: {}".format(acc, f1, presion, recall))
    print(f"final test classification report: \n {classification_report(all_labels, all_preds)}")


def main():
    passer = argparse.ArgumentParser()
    passer.add_argument('--batch_size', type=int, default=768)
    passer.add_argument('--lr', type=float, default=2e-4)
    passer.add_argument('--epochs', type=int, default=20)
    passer.add_argument('--save_path', type=str, default='./out_models/')
    passer.add_argument('--model_name', choices=["baseline_bert", "cnn", "lstm"], default="cnn", type=str,
                        help="模型名")

    args = passer.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists('./log'):
        os.mkdir('./log')

    # 设置随机种子
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)
    # 加载数据
    print("开始加载数据")
    t1 = time.time()
    train_loader, dev_loader, test_loader = get_dataloader(args.batch_size)
    print(f"加载数据完成, 耗时{time.time() - t1}s")
    # 加载模型,vocab_size等于BertTokenizer的vocab_size
    print("开始加载模型")
    vocab_size = 30522
    model = CNNModel(vocab_size)
    # 设置GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # 训练模型
    print("开始训练模型")
    t2 = time.time()
    for epoch in range(args.epochs):
        train(args, model, train_loader, dev_loader, optimizer, device, epoch)
    print(f"训练模型完成, 耗时{time.time() - t2}s")
    # 预测
    save_path = './out_models/'
    if os.path.exists(save_path + f'best_{args.model_name}_model.pt'):
        model.load_state_dict(torch.load(save_path + f'best_{args.model_name}_model.pt'))
    # model.load_state_dict(torch.load(save_path + f'best_baseline_bert_model.pt'))
    print("开始预测")
    predict(model, test_loader, device)
    print(f"预测完成,总耗时{time.time() - t1}s")


if __name__ == '__main__':
    main()
