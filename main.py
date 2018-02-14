from utils import DataLoader
from lm import TrigramLM

import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    data_loader = DataLoader()
    sentences = data_loader.load_data(path='data/brown', count=1000)
    np.random.shuffle(sentences)
    X_train, X_test = train_test_split(sentences, test_size=0.1)
    # print X_train[:10]
    trigram_lm = TrigramLM()
    trigram_lm.train(X_train)
    print(trigram_lm.bigram_counter.count(['the', 'city']))
