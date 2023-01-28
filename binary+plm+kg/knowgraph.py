class KnowledgeGraph(object):
    def __init__(self, args, predicate=True):
        '''

        :param args:
        :param predicate: 是否在三元组中包含谓词
        '''
        self.spo_path = args.kg_path
        self.predicate = predicate
        # 创建查找表
        self.lookup_table = self._create_lookup_table()

    def _create_lookup_table(self):
        '''
        创建查找表
        :return:
        '''
        lookup_table = {}
        print("[KnowledgeGraph] Loading spo from {}".format(self.spo_path))
        with open(self.spo_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    subj, pred, obje = line.strip().split("\t")
                    subj, pred, obje = subj.lower(), pred.lower(), obje.lower()
                except:
                    print("[KnowledgeGraph] Bad spo:", line)
                if self.predicate:
                    value = pred + ' ' + obje
                else:
                    value = obje
                if subj in lookup_table.keys():
                    lookup_table[subj].add(value)
                else:
                    lookup_table[subj] = set([value])
        return lookup_table

    def has(self, text):
        return text in self.lookup_table.keys()

    def get(self, text):
        '''
        逐词查找文本中最长 token 对应的三元组，返回查找结果组成的 list
        考虑一对多的情况
        '''
        # print(text)
        subjs, results = [], []
        words = text.split(' ')
        i, j = 0, 0
        while i < len(words):
            if self.has(words[i]) is True:
                lookup_word = words[i]
                j = i + 1
                while j < len(words) and self.has(lookup_word + ' ' + words[j]) is True:
                    lookup_word = lookup_word + ' ' + words[j]
                    j += 1
                results.append(self.lookup_table[lookup_word])
                subjs.append(lookup_word)
                i = j
            else:
                i += 1
        return subjs, results
