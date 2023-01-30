import spacy
import json

def parse(args):
    postag_set = set()
    pos_set = set()
    postag_dict = {}
    pos_dict = {}

    print("Building an English pipeline...")
    nlp = spacy.load('en_core_web_sm')
    for data_path in [args.train_path, args.test_path, args.dev_path]:
        with open(data_path, 'r') as f:
            lines = f.readlines()
            for line_id, line in enumerate(lines):
                print("{}/{}".format(line_id, len(lines)))
                if line_id == 0:
                    continue
                _, text = line.split('\t')
                doc = nlp(text)
                poses, postags = [], []
                for pos_id, token in enumerate(doc):
                    poses.append(pos_id)
                    postags.append(token.tag_)
                    pos_set.add(pos_id)
                    postag_set.add(token.tag_)
                postag_dict[line_id] = postags
                pos_dict[line_id] = poses

        json_dict = json.dumps(postag_dict)
        with open(data_path.replace('.tsv', '_postag.json'), 'w') as json_file:
            json_file.write(json_dict)
        json_dict = json.dumps(pos_dict)
        with open(data_path.replace('.tsv', '_pos.json'), 'w') as json_file:
            json_file.write(json_dict)

    with open(args.dataset_path+'postag_vocab.txt', 'w') as f:
        for x in postag_set:
            f.write(str(x)+'\n')
    with open(args.dataset_path+'pos_vocab.txt', 'w') as f:
        for x in pos_set:
            f.write(str(x)+'\n')
