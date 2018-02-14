from utils.counter import NGramCounter
from utils import counter
import numpy as  np


class LanguageModel(object):
    START = '<s>'
    STOP = '</s>'

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

    def logprob(self, sentence):
        raise NotImplementedError

    def predict(self, k=1):
        raise NotImplementedError

    def train(self, data):
        raise NotImplementedError

    def evaluate(self, data):
        raise NotImplementedError


class TrigramLM(LanguageModel):

    def logprob(self, sentence):
        ans = 0
        for i, word in enumerate(sentence):
            ans += self._logprob(word[i], [word[i - 1], word[i - 2]])
        return ans

    def train(self, data):
        num_success = 0
        num_error = 0
        for words in data:
            try:
                words = np.append([[self.START, 'st']], words, axis=0)
                words = np.append([[self.START, 'st']], words, axis=0)
                words = np.append(words, [[self.STOP, 'ed']], axis=0)
                for i in range(2, len(words)):
                    self.unigram_counter.add(words[i:i+1,0])
                    self.bigram_counter.add(words[i-1:i+1,0])
                    self.trigram_counter.add(words[i-2:i+1,0])
                num_success += 1
            except:
                num_error += 1
                continue
        print("Finish adding ngrams :: {} success :: {} error".format(num_success, num_error))

    def _logprob(self, param, param1):
        pass


    def _missing_prob_mass(self, ngram):
        # return self._count_star(ngram) * 1.0 / self.bigram_counter.count(ngram[:-1])
        if len(ngram) == counter.UNIGRAM:
            ans = 0
            cnt = self.unigram_counter.count(ngram)
            for word in self.bigram_counter.ngram_with_prefix(' '.join(ngram) + ' '):
                ans += 1.0 * self._count_star(word.split()) / cnt
            return 1.0 - ans
        if len(ngram) == counter.BIGRAM:
            ans = 0
            cnt = self.bigram_counter.counter(ngram)
            for word in self.trigram_counter.ngram_with_prefix(' '.join(ngram) + ' '):
                ans += 1.0 * self._count_star(word.split()) / cnt
            return 1.0 - ans

    def _count_star(self, ngram):
        if len(ngram) == counter.UNIGRAM:
            return self.unigram_counter.count(ngram) - self.theta
        if len(ngram) == counter.BIGRAM:
            return self.bigram_counter.count(ngram) - self.theta
        if len(ngram) == counter.TRIGRAM:
            return self.trigram_counter.count(ngram) - self.theta

    def _prob(self, ngram):
        if len(ngram) == counter.BIGRAM:
            if self.bigram_counter.count(ngram) > 0:
                return self._count_star(ngram)
            else:
                return self._missing_prob_mass(ngram) * (self._ml_estimate(ngram) / (1.0))

    def _ml_estimate(self, ngram):
        '''
        maximum likelihood estimation
        :param ngram: list of words
        :return: probability of ngram
        '''
        if len(ngram) == counter.UNIGRAM:
            return self.unigram_counter.count(ngram) * 1.0 / self.unigram_counter.total()
        if len(ngram) == counter.BIGRAM:
            return self.bigram_counter.counter(ngram) * 1.0 / self.unigram_counter.counter(ngram[:-1])
        if len(ngram) == counter.TRIGRAM:
            return self.trigram_counter.counter(ngram) * 1.0 / self.bigram_counter.counter(ngram[:-1])
