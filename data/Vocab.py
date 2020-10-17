from collections import Counter
import numpy as np
import re

class Vocab(object):
    PAD, UNK = 0, 1
    maskedwordUNK = 0
    def __init__(self, word_counter, masked_word_counter, config):
        self._id2word = ['<pad>', '<unk>']
        self._wordid2freq = [10000, 10000]
        self._id2extword = ['<pad>', '<unk>']
        self._id2maskedword = ['<unk>']

        for word, count in word_counter.most_common():
            if count > config.min_occur_count:
                self._id2word.append(word)
                self._wordid2freq.append(count)

        for masked_word, count in masked_word_counter.most_common():
            if count > config.min_masked_occur_count:
                self._id2maskedword.append(masked_word)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        self._maskedword2id = reverse(self._id2maskedword)
        if len(self._maskedword2id) != len(self._id2maskedword):
            print("serious bug: masked word dumplicated, please check!")

        #print("Masked word info: ", self._id2maskedword)
        print("Vocab info: #words %d, #masked word size: %d"
              % (self.vocab_size, self.maskedword_size))

    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                self._id2extword.append(values[0])
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        if len(self._extword2id) != len(self._id2extword):
            print("serious bug: extern words dumplicated, please check!")

        return embeddings

    def create_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword) - word_count
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if self._extword2id.get(values[0], self.UNK) != index:
                    print("Broken vocab or error embedding file, please check!")
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        embeddings = embeddings / np.std(embeddings)

        return embeddings


    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def wordid2freq(self, xs):
        if isinstance(xs, list):
            return [self._wordid2freq[x] for x in xs]
        return self._wordid2freq[xs]

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.UNK) for x in xs]
        return self._extword2id.get(xs, self.UNK)

    def id2extword(self, xs):
        if isinstance(xs, list):
            return [self._id2extword[x] for x in xs]
        return self._id2extword[xs]

    def maskedword2id(self, xs):
        if isinstance(xs, list):
            return [self._maskedword2id.get(x, self.maskedwordUNK) for x in xs]
        return self._maskedword2id.get(xs, self.maskedwordUNK)

    def id2maskedword(self, xs):
        if isinstance(xs, list):
            return [self._id2maskedword[x] for x in xs]
        return self._id2maskedword[xs]


    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def extvocab_size(self):
        return len(self._id2extword)

    @property
    def maskedword_size(self):
        return len(self._maskedword2id)

def normalize_to_lowerwithdigit(str):
    str = str.lower()
    str = re.sub(r'\d', '0', str) ### replace digit 2 zero
    return str

def creatVocab(train_data, config, masked_word_counter):
    word_counter = Counter()
    for inst in train_data:
        for sentence in inst.masked_EDUs:
            for word in sentence:
                word_counter[word] += 1

    return Vocab(word_counter, masked_word_counter, config)

def load_conj(conjunction_file):
    conj_counter = Counter()
    with open(conjunction_file, mode='r', encoding='utf8') as inf:
        for line in inf.readlines():
            line = line.strip()
            index = line.find(' ')
            if index == -1:
                conj_counter[line] += 1
    return conj_counter
