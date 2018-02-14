from utils.counter import NGramCounter
from utils import counter
import numpy as  np


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
            self.bigram_counter = NGramCounter(counter.UNIGRAM)
        if self.mode == counter.TRIGRAM:
            self.unigram_counter = NGramCounter(counter.UNIGRAM)
            self.bigram_counter = NGramCounter(counter.UNIGRAM)
            self.trigram_counter = NGramCounter(counter.UNIGRAM)

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
    def __init__(self, smoothing='katz', theta=0.1):
        LanguageModel.__init__(self, counter.TRIGRAM, smoothing=smoothing, theta=theta)
        self.vocab = set()
    def train(self, data):
        def train(self, data):
            num_success = 0
            num_error = 0
            for words in data:
                try:
                    words = np.append([[LanguageModel.START, 'st']], words, axis=0)
                    words = np.append([[LanguageModel.START, 'st']], words, axis=0)
                    words = np.append(words, [[LanguageModel.STOP, 'ed']], axis=0)
                    for i in range(2, len(words)):
                        self.unigram_counter.add(words[i:i + 1, 0])
                        self.bigram_counter.add(words[i - 1:i + 1, 0])
                        self.trigram_counter.add(words[i - 2:i + 1, 0])
                    num_success += 1
                except:
                    num_error += 1
                    continue
            print("Finish adding ngrams :: {} success :: {} error".format(num_success, num_error))

    def logprob(self, sentence):
        ans = 0
        for i, word in enumerate(sentence):
            ans += self._logprob(word[i], [word[i - 1], word[i - 2]])
        return ans

    def _logprob(self, param, param1):
        pass



    def _missing_prob_mass(self, ngram):
        # return self._discounted_prob(ngram) * 1.0 / self.bigram_counter.count(ngram[:-1])
        ans = 0
        if len(ngram) == counter.UNIGRAM:
            cnt = self.unigram_counter.count(ngram)
        if len(ngram) == counter.BIGRAM:
            cnt = self.bigram_counter.counter(ngram)

        for word in self.bigram_counter.ngram_with_prefix(' '.join(ngram) + ' '):
            ans += 1.0 * self._discounted_prob(word.split()) / cnt
        return 1.0 - ans

    def _discounted_prob(self, ngram):
        if len(ngram) == counter.UNIGRAM:
            return self.unigram_counter.count(ngram) - self.theta
        if len(ngram) == counter.BIGRAM:
            return self.bigram_counter.count(ngram) - self.theta
        if len(ngram) == counter.TRIGRAM:
            return self.trigram_counter.count(ngram) - self.theta

    def _backoff_weight(self, ngram):
        pass

    def _katz_backoff_prob(self, ngram):
        '''
            calculated Katz backoff
        '''
        #in set A

        if len(ngram) == counter.UNIGRAM:
            return self._ml_estimate(ngram)
        if self._exist(ngram) > 0: 
            return self._discounted_prob(ngram)
        #in set B
        else:
            return self._backoff_weight(ngram[:-1]) * self._katz_backoff_prob(ngram[1:])

    def _ml_estimate(self, ngram):
        '''
            Calculate maximum likelihood estimation
        '''
        if len(ngram) == counter.UNIGRAM:
            return self.unigram_counter.count(ngram) * 1.0 / self.unigram_counter.total()
        if len(ngram) == counter.BIGRAM:
            return self.bigram_counter.count(ngram) * 1.0 / self.unigram_counter.count(ngram[:-1])
        if len(ngram) == counter.TRIGRAM:
            return self.trigram_counter.count(ngram) * 1.0 / self.bigram_counter.count(ngram[:-1])
    def _add(self, ngram):
        if len(ngram) == 1:
            self.vocab.add(ngram[0])
            self.unigram_counter.add(ngram)
        if len(ngram) == 2:
            self.bigram_counter.add(ngram)
        if len(ngram) == 3:
            self.trigram_counter.add(ngram)
    def _exist(self, ngram):
        full_counter = None
        if len(ngram) == counter.UNIGRAM:
            full_counter = self.unigram_counter
        if len(ngram) == counter.BIGRAM:
            full_counter = self.bigram_counter
        if len(ngram) == counter.TRIGRAM:
            full_counter = self.trigram_counter
        return full_counter.count(ngram) > 0
