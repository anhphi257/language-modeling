from utils import Trie
import utils

import logging

logging.getLogger().setLevel(logging.INFO)
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
        else:
            raise TypeError("Only support unigram, bigram, trigram")

    def add(self, words):
        # logging.info("add {}".format(words))
        if len(words) in [UNIGRAM, BIGRAM, TRIGRAM]:
            self.counter.add(utils.merge(words))

        else:
            raise TypeError("Only support unigram, bigram, trigram")

    def count(self, words):
        ans = self.counter.count(utils.merge(words))
        # logging.info("counting: {} : {}".format(' '.join(words), ans))
        return ans

    def total(self):
        return self.counter.total()

    def ngram_with_prefix(self, ngram):
        return self.counter.start_with_prefix(' '.join(ngram))
