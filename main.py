from utils import DataLoader
from lm import TrigramLM
import logging
import pickle

logging.getLogger().setLevel(logging.INFO)
import numpy as np
from sklearn.model_selection import train_test_split

def readData(count = 1000):
    data_loader = DataLoader()
    sentences = data_loader.load_data(path='data/brown', count=count)
    np.random.shuffle(sentences)
    X_train, X_test = train_test_split(sentences, test_size=0.1)
    f_train = open('data/train.pkl', 'wb')
    f_test = open('data/test.pkl', 'wb')
    for x in X_train:
        pickle.dump(x, f_train)
    for x in X_test:
        pickle.dump(x, f_test)
    del X_train
    del X_test
    del sentences
    f_train.close()
    f_test.close()

if __name__ == '__main__':
    readData(1000)
    trigram_lm = TrigramLM()
    trigram_lm.train('data/train.pkl')

    # logging.info("vocab_size: {}".format(len(trigram_lm.vocab)))
    # print trigram_lm.unigram_counter.count([])
    # print trigram_lm.vocab
    # print trigram_lm.logprob('who is butcher')
    # print trigram_lm.logprob('c a b')
