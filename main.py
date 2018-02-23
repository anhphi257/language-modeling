from utils import DataLoader
from lm import TrigramLM
import logging

logging.getLogger().setLevel(logging.INFO)
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data_loader = DataLoader()
    sentences = data_loader.load_data(path='data/brown', count=1000)
    np.random.shuffle(sentences)
    X_train, X_test = train_test_split(sentences, test_size=0.1)
    # print X_train[:10]
    # s1 = [[['a', 'as'], ['b', 'as'], ['c', 'as']],
    #       [['b', 'as'], ['a', 'as'], ['c', 'as']]]

    trigram_lm = TrigramLM()

    trigram_lm.train(X_train)
    del X_train
    # logging.info("vocab_size: {}".format(len(trigram_lm.vocab)))
    # print trigram_lm.unigram_counter.count([])
    # print trigram_lm.vocab
    print trigram_lm.logprob('who is butcher')
    # print trigram_lm.logprob('c a b')
