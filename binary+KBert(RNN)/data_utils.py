import json
import random
import pandas as pd
import os

def json_to_spo(json_path, spo_path):
    with open(json_path, 'r', encoding='UTF-8') as f1:
        dict = json.load(f1)
        with open(spo_path, 'w', encoding='UTF-8') as f2:
            for k, v in dict.items():
                for l in v:
                    f2.write(l[0]+'\t'+l[1]+'\t'+l[2]+'\n')

def split_train_test(data_path, filename, ratio=0.7):
    data_df = pd.read_csv(data_path+filename, sep='\t', header=0)
    all_data = pd.DataFrame(columns=['label', 'text_a'])

    label_map = {'not_hate':0, 'implicit_hate':1, 'explicit_hate':2}
    # 将标签映射为数字
    all_data['label'] = [label_map[x] for x in data_df['class']]
    all_data['text_a'] = data_df['post']

    train_data = all_data.sample(frac=ratio)
    all_data = all_data[~all_data.index.isin(train_data.index)]
    test_data = all_data.sample(frac=0.5)
    dev_data = all_data[~all_data.index.isin(test_data.index)]
    train_data.to_csv(data_path+"train.tsv", sep='\t', index=False)
    test_data.to_csv(data_path+"test.tsv", sep='\t', index=False)
    dev_data.to_csv(data_path+"dev.tsv", sep='\t', index=False)

def build_pretrain_data(path = 'data/pre_train', save_path = 'data/pretrain_data.txt'):
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    s = []
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            f = open(path + "/" + file);  # 打开文件
            iter_f = iter(f);  # 创建迭代器
            str = ""
            for line in iter_f:  # 遍历文件，一行行遍历，读取文本
                str = str + line
            s.append(str)  # 每个文件的文本存到list中
    with open(save_path, "w") as f:
        for line in s:
            f.write(line+' ')

def handle_spo(spo_path):
    '''
    对 spo 的一些初始化处理操作
    '''
    with open(spo_path, "r") as f1, open(spo_path+'1.spo', 'w') as f2:
        lines = f1.readlines()
        for line_id, line in enumerate(lines):
            # print("{}/{}".format(line_id, len(lines)))
            print(line)
            a, b, c = line.split('\t')
            a, b, c = a.strip(), b.strip(), c.strip()
            f2.write(a+'\t'+b+'\t'+c+'\n')


# if __name__ == '__main__':
    # json_to_spo("data/triples.json", "data/triples.spo")
    # split_train_test("data/implicit_hate_v1_stg1_posts/", "implicit_hate_v1_stg1_posts.tsv")
    # build_pretrain_data(path = 'data/pre_train')

    # handle_spo('brain/kgs/hate_brains.spo')