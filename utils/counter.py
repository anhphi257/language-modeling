from utils import Trie
import numpy as np

UNIGRAM = 1
BIGRAM = 2
TRIGRAM = 3


class Counter(object):
    @classmethod
    def count(cls, word):
        raise NotImplementedError("Please subclass this class")

    @classmethod
    def add(cls, word):
        raise NotImplementedError("Please subclass this class")


class NGramCounter(Counter):
    def __init__(self, type=TRIGRAM):
        if type in [UNIGRAM, BIGRAM, TRIGRAM]:
            self.mode = type
            self.counter = Trie()
            self.adj_words = dict()
        else:
            raise TypeError("Only support unigram, bigram, trigram")

    def add(self, words):
        if len(words) in [UNIGRAM, BIGRAM, TRIGRAM]:
            self.counter.add(' '.join(words))
            if len(words) > 2:
                key = ' '.join(words[:-1])
                value = words[-1]
        else:
            raise TypeError("Only support unigram, bigram, trigram")

    def count(self, words):
        return self.counter.count(' '.join(words))

    def total(self):
        return self.counter.total()

    def ngram_with_prefix(self, ngram):
        return self.counter.start_with_prefix(' '.join(ngram))
