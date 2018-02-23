from __future__ import division
from utils.counter import NGramCounter
from utils import counter
import utils
import numpy as np
import logging
import time
logging.getLogger().setLevel(logging.INFO)


class LanguageModel(object):
    START = '<s>'
    STOP = '</s>'
    UNK = '<UNK>'

    def __init__(self, mode=counter.TRIGRAM, smoothing='katz', theta=0.1):
        self.mode = mode
        self.smoothing = smoothing
        self.theta = theta
        if self.mode == counter.UNIGRAM:
            self.unigram_counter = NGramCounter(counter.UNIGRAM)
        if self.mode == counter.BIGRAM:
            self.unigram_counter = NGramCounter(counter.UNIGRAM)
            self.bigram_counter = NGramCounter(counter.BIGRAM)
        if self.mode == counter.TRIGRAM:
            self.unigram_counter = NGramCounter(counter.UNIGRAM)
            self.bigram_counter = NGramCounter(counter.BIGRAM)
            self.trigram_counter = NGramCounter(counter.TRIGRAM)

    @classmethod
    def logprob(cls, sentence):
        raise NotImplementedError

    @classmethod
    def predict(cls, k=1):
        raise NotImplementedError

    @classmethod
    def train(cls, data):
        raise NotImplementedError

    @classmethod
    def evaluate(cls, data):
        raise NotImplementedError


class TrigramLM(LanguageModel):
    def __init__(self, smoothing='katz', theta=0.5):
        LanguageModel.__init__(self, counter.TRIGRAM, smoothing=smoothing, theta=theta)
        self.vocab = set()
        self.adj_words = dict()

    def _add(self, ngram):
        # print ngram
        if len(ngram) == 1:
            self.vocab.add(ngram[0])
            self.unigram_counter.add(ngram)
        if len(ngram) == 2:
            self.bigram_counter.add(ngram)
        if len(ngram) == 3:
            self.trigram_counter.add(ngram)
        if len(ngram) > 2:
            key = ' '.join(ngram[:-1])
            value = ngram[-1]
            try:
                self.adj_words[key].add(value)
            except:
                self.adj_words[key] = set()
                self.adj_words[key].add(value)

    def _exist(self, ngram):
        full_counter = None
        if len(ngram) == counter.UNIGRAM:
            full_counter = self.unigram_counter
        if len(ngram) == counter.BIGRAM:
            full_counter = self.bigram_counter
        if len(ngram) == counter.TRIGRAM:
            full_counter = self.trigram_counter
        return full_counter.count(ngram) > 0

    def train(self, data):
        num_success = 0
        num_error = 0
        for words in data:
            try:
                # print words
                # words = np.append([[LanguageModel.START, 'st']], words, axis=0)
                words = np.append([[LanguageModel.START, 'st']], words, axis=0)
                # words = np.append(words, [[LanguageModel.STOP, 'ed']], axis=0)
                words = np.append(words, [[LanguageModel.STOP, 'ed']], axis=0)
                # print words

                for i in range(0, len(words)):

                    self._add(words[i:i + 1, 0])  # add unigram
                    if i < len(words) - 1:
                        self._add(words[i:i + 2, 0])
                    if i < len(words) - 2:
                        self._add(words[i:i + 3, 0])
                num_success += 1
            except:
                num_error += 1
                continue
        print("Finish adding ngrams :: {} success :: {} error".format(num_success, num_error))

    def logprob(self, sentence):
        sentence = '{} {} {}'.format(LanguageModel.START, sentence, LanguageModel.STOP).split()
        ans = 0
        for i in range(2, len(sentence)):
            start_time = time.time()
            prob = self._katz_backoff_prob(sentence[i - 2:i + 1])
            end_time = time.time()
            logging.info("{} : {} in {}s".format(sentence[i - 2:i + 1], prob, end_time - start_time))

            ans += np.log(prob)
        return ans

    def evaluate(self, data):
        ans = 0

        for sentence in data:
            ans += self.logprob(sentence)
        return ans

    def _discounted_prob(self, ngram):
        if len(ngram) == counter.UNIGRAM:
            return (self.unigram_counter.count(ngram) * 1.0 - self.theta) / self.unigram_counter.total()
        if len(ngram) == counter.BIGRAM:
            return (self.bigram_counter.count(ngram) * 1.0 - self.theta) / self.unigram_counter.count(ngram[:-1])
        if len(ngram) == counter.TRIGRAM:
            return (self.trigram_counter.count(ngram) * 1.0 - self.theta) / self.bigram_counter.count(ngram[:-1])

    def _set_A(self, ngram):
        try:
            return self.adj_words[utils.merge(ngram)]
        except:
            return set()

    def _set_B(self, ngram):
        return self.vocab.difference(self._set_A(ngram))

    def _katz_backoff_prob(self, ngram, mle=False):
        '''
            calculated Katz backoff
        '''
        # logging.info("Calculate backoff of \'{}\'".format(' '.join(ngram)))

        if len(ngram) == 1:
            # if mle:
            return self._ml_estimate(ngram)
            # else:
            #     return self._discounted_prob(ngram)
        if self._exist(ngram) > 0:
            # print  self._discounted_prob(ngram)
            return self._discounted_prob(ngram)
        # in set B
        else:
            # logging.info("{}, {}".format(ngram[:-1], ngram[1:]))
            # print self._backoff_weight(ngram[:-1]) * self._katz_backoff_prob(ngram[1:])
            return self._backoff_weight(ngram[:-1]) * self._katz_backoff_prob(ngram[1:])

    def _ml_estimate(self, ngram):
        '''
            Calculate maximum likelihood estimation
        '''
        # logging.info("ML estimaste: {}".format(ngram))
        if len(ngram) == counter.UNIGRAM:
            return self.unigram_counter.count(ngram) * 1.0 / len(self.vocab)
        if len(ngram) == counter.BIGRAM:
            return self.bigram_counter.count(ngram) * 1.0 / self.unigram_counter.count(ngram[:-1])
        if len(ngram) == counter.TRIGRAM:
            return self.trigram_counter.count(ngram) * 1.0 / self.bigram_counter.count(ngram[:-1])

    def _backoff_weight(self, ngram):

        set_a = self._set_A(ngram)
        set_b = self._set_B(ngram)
        # ngram = [w_(j-2), w_(j-1)]
        if len(ngram) == 2:
            alpha = 1
            for w in set_a:
                alpha -= self._discounted_prob(ngram + [w])
            beta = 0
            for w in set_b:
                beta += self._katz_backoff_prob([ngram[1], w])
            return alpha / beta
        # ngram = [w_(j-1)]
        if len(ngram) == 1:
            alpha = 1
            for w in set_a:
                alpha -= self._discounted_prob(ngram + [w])
            beta = 0
            for w in set_b:
                beta += self._ml_estimate([w])
            return alpha / beta
