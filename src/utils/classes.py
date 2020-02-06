# This file contains different user-defined classes


class Vocabulary(object):
    """Vocabulary class"""
    def __init__(self):
        super(Vocabulary, self).__init__()
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.word2count = {}
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.size = 4  # count the special tokens above

    def add_sentence(self, sentence):
        for word in sentence.strip().split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.size
            self.word2count[word] = 1
            self.index2word[self.size] = word
            self.size += 1
        else:
            self.word2count[word] += 1

    def sentence2index(self, sentence):
        indexes = []
        for w in sentence.split():
            try:
                indexes.append(self.word2index[w])
            except KeyError as e:  # handle OOV
                indexes.append(self.word2index['<UNK>'])
        return indexes

    def index2sentence(self, indexes):
        return [self.index2word[i] for i in indexes]


class Dataset(object):
    """docstring for Dataset"""
    def __init__(self, arg):
        super(Dataset, self).__init__()
        self.arg = arg

