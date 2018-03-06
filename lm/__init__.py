from __future__ import division
from utils.counter import NGramCounter
from utils import counter
import utils
import numpy as np
import logging
import time
import pickle
import time
logging.getLogger().setLevel(logging.INFO)


class LanguageModel(object):
    START = '<s>'
    STOP = '</s>'
    UNKNOWN = '<unk>'

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
    PROB = 0
    BOW = 1

    def __init__(self, smoothing='katz', theta=0.5):
        LanguageModel.__init__(self, counter.TRIGRAM, smoothing=smoothing, theta=theta)
        self.unigram_vocab = set()
        self.bigram_vocab = set()
        self.trigram_vocab = set()
        self.adj_words = dict()
        self.unigram_arpa = dict()
        self.bigram_arpa = dict()
        self.trigram_arpa = dict()

    def _add(self, ngram):
        # print ngram
        if len(ngram) == 1:
            self.unigram_vocab.add(ngram[0])
            self.unigram_counter.add(ngram)
        if len(ngram) == 2:
            self.bigram_vocab.add(utils.merge(ngram))
            self.bigram_counter.add(ngram)
        if len(ngram) == 3:
            self.trigram_vocab.add(utils.merge(ngram))
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

    def train(self, path):
        self._train_add_gram(path)
        self._train_unigram()
        self._train_bigram()
        self._train_trigram()
        self._train_save_arpa()

    def _train_add_gram(self, path):
        start_time = time.time()
        num_success = 0
        num_error = 0
        f = open(path, 'rb')
        while True:
            try:
                words = pickle.load(f)
            except EOFError:
                break
            try:
                words = np.append([[LanguageModel.START, 'st'],[LanguageModel.START, 'st']], words, axis=0)
                words = np.append(words, [[LanguageModel.STOP, 'ed']], axis=0)
                for i in range(1, len(words)):
                    # print("{} {} {}".format(words[i:i+1, 0], words[i-1:i+1, 0], words[i-2:i+1, 0]))
                    self._add(words[i:i+1, 0])
                    self._add(words[i-1:i+1, 0])
                    if i-2 >= 0:
                        self._add(words[i-2:i+1, 0])
                num_success += 1
                del words
            except:
                num_error += 1
                del words
                continue
        elapsed_time = time.time() - start_time
        logging.info("Finish adding ngrams :: {} success :: {} error :: took {}s".format(num_success, num_error, elapsed_time))
        logging.info("{} 1-gram :: {} 2-grams :: {} 3-grams".format(len(self.unigram_vocab), len(self.bigram_vocab), len(self.trigram_vocab)))

    def _train_unigram(self):
        start_time = time.time()
        num_success = 0
        num_error = 0
        for word in self.unigram_vocab:
            try:
                self.unigram_arpa[word] = [10.0**-99, None]
                if word != LanguageModel.START:
                    self.unigram_arpa[word][self.PROB] = self._ml_estimate([word])
                count_word = self.unigram_counter.count([word])
                if word != LanguageModel.STOP:
                    bow_top = 0.0
                    bow_bottom = 0.0
                    for word2 in self.unigram_vocab:
                        if word2 == LanguageModel.START:
                            continue
                        count_word_word2 = self.bigram_counter.count([word, word2])
                        if count_word_word2 > 0:
                            bow_top += self._discount(count_word_word2)
                        else:
                            bow_bottom += self._ml_estimate([word2])
                    bow_top = 1.0 - bow_top / count_word
                    self.unigram_arpa[word][self.BOW] = bow_top / bow_bottom
                num_success += 1
            except:
                num_error += 1
        elapsed_time = time.time() - start_time
        logging.info("Finish calculating 1-gram :: {} success :: {} error :: took {}s".format(num_success, num_error, elapsed_time))

    def _train_bigram(self):
        start_time = time.time()
        num_success = 0
        num_error = 0
        for sentence in self.bigram_vocab:
            try:
                bigram = sentence.strip().split()
                word = bigram[0]
                word2 = bigram[1]
                if word == LanguageModel.START and word2 == LanguageModel.START:
                    continue
                self.bigram_arpa[sentence] = [None, None]
                count_word_word2 = self.bigram_counter.count(bigram)
                if count_word_word2 > 0:
                    self.bigram_arpa[sentence][self.PROB] = self._discounted_prob(bigram)
                else:
                    self.bigram_arpa[sentence][self.PROB] = self.unigram_arpa[word][self.BOW]*self._ml_estimate([word2])
                if word2 != LanguageModel.STOP:
                    bow_top = 0.0
                    bow_bottom = 0.0
                    for word3 in self.unigram_vocab:
                        count_word_word2_word3 = self.trigram_counter.count([word, word2, word3])
                        if count_word_word2_word3 > 0:
                            bow_top += self._discount(count_word_word2_word3)
                        else:
                            count_word2_word3 = self.bigram_counter.count([word2, word3])
                            if count_word2_word3 > 0:
                                bow_bottom += self._discounted_prob([word2, word3])
                            else:
                                bow_bottom += self.unigram_arpa[word2][self.BOW] * self.unigram_arpa[word3][self.PROB]
                    bow_top = 1.0 - bow_top / count_word_word2
                    if np.abs(bow_top-bow_bottom) > 0.000001:
                        self.bigram_arpa[sentence][self.BOW] = bow_top / bow_bottom
                num_success += 1
            except:
                num_error += 1
        elapsed_time = time.time() - start_time
        logging.info("Finish calculating 2-gram :: {} success :: {} error :: took {}s".format(num_success, num_error, elapsed_time))

    def _train_trigram(self):
        start_time = time.time()
        num_success = 0
        num_error = 0
        for sentence in self.trigram_vocab:
            try:
                trigram = sentence.strip().split()
                word = trigram[0]
                word2 = trigram[1]
                word3 = trigram[2]
                self.trigram_arpa[sentence] = [None, None]
                count_word_word2_word3 = self.trigram_counter.count(trigram)
                if count_word_word2_word3 > 0:
                    self.trigram_arpa[sentence][self.PROB] = self._discounted_prob(trigram)
                else:
                    self.trigram_arpa[sentence][self.PROB] = self.bigram_arpa[utils.merge([word, word2])][self.BOW]
                    count_word2_word3 = self.bigram_counter.count([word2, word3])
                    if count_word2_word3 > 0:
                        self.trigram_arpa[sentence][self.PROB] *= self.bigram_arpa[utils.merge([word2, word3])][self.PROB]
                    else:
                        self.trigram_arpa[sentence][self.PROB] *= self.unigram_arpa[word2][self.BOW] * self.unigram_arpa[word3][self.PROB]
                num_success += 1
            except:
                num_error += 1
        elapsed_time = time.time() - start_time
        logging.info("Finish calculating 3-gram :: {} success :: {} error :: took {}s".format(num_success, num_error, elapsed_time))

    def _train_save_arpa(self, path='data/lm.arpa'):
        start_time = time.time()
        with open(path, 'w') as f:
            f.write('\\data\\\n')
            f.write('ngram 1={}\n'.format(len(self.unigram_arpa)))
            f.write('ngram 2={}\n'.format(len(self.bigram_arpa)))
            f.write('ngram 3={}\n'.format(len(self.trigram_arpa)))
            f.write('\n')
            f.write('\\1-grams:\n')
            for word,info in sorted(self.unigram_arpa.items()):
                if info[self.BOW] is None:
                    f.write("{:.5f}\t{}\n".format(np.log10(info[self.PROB]), word))
                else:
                    f.write("{:.5f}\t{}\t{:.5f}\n".format(np.log10(info[self.PROB]), word, np.log10(info[self.BOW])))
            f.write('\n')
            f.write('\\2-grams:\n')
            for word,info in sorted(self.bigram_arpa.items()):
                if info[self.PROB] is not None:
                    if info[self.BOW] is None:
                        f.write("{:.5f}\t{}\n".format(np.log10(info[self.PROB]), word))
                    else:
                        f.write("{:.5f}\t{}\t{:.5f}\n".format(np.log10(info[self.PROB]), word, np.log10(info[self.BOW])))
            f.write('\n')
            f.write('\\3-grams:\n')
            for word,info in sorted(self.trigram_arpa.items()):
                if info[self.PROB] is not None:
                    f.write("{:.5f}\t{}\n".format(np.log10(info[self.PROB]), word))
            f.write('\n')
            f.write('\\end\\\n')
            f.close()
        elapsed_time = time.time() - start_time
        logging.info("Finish saving lm: {}".format(path))


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

    def _discount(self, count):
        return count - self.theta

    def _discounted_count(self, ngram):
        if len(ngram) == counter.UNIGRAM:
            return (self.unigram_counter.count(ngram) * 1.0 - self.theta)
        if len(ngram) == counter.BIGRAM:
            return (self.bigram_counter.count(ngram) * 1.0 - self.theta)
        if len(ngram) == counter.TRIGRAM:
            return (self.trigram_counter.count(ngram) * 1.0 - self.theta)

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
            return self.unigram_counter.count(ngram) * 1.0 / self.unigram_counter.total()
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
