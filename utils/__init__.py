import os
import re
import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)


class Node:
    def __init__(self, label=None, freq=None, data=None):
        self.label = label
        self.freq = freq
        self.data = data
        self.children = dict()

    def addChild(self, key, data=None):
        if not isinstance(key, Node):
            self.children[key] = Node(key, data)
        else:
            self.children[key.label] = key

    def __getitem__(self, key):
        return self.children[key]


class Trie:
    def __init__(self):
        self.head = Node()
        self.sum = 0

    def __getitem__(self, key):
        return self.head.children[key]

    def add(self, word):
        # print word
        self.sum += 1
        current_node = self.head
        word_finished = True

        for i in range(len(word)):
            if word[i] in current_node.children:
                current_node = current_node.children[word[i]]
            else:
                word_finished = False
                break

        # For ever new letter, create a new child node
        if not word_finished:
            while i < len(word):
                current_node.addChild(word[i])
                current_node = current_node.children[word[i]]
                i += 1

        # Let's store the full word at the end node so we don't need to
        # travel back up the tree to reconstruct the word
        if current_node.freq is None:
            current_node.freq = 0
        current_node.freq += 1
        current_node.data = word

    def has_word(self, word):
        if word == '':
            return False
        if word == None:
            raise ValueError('Trie.has_word requires a not-Null string')

        # Start at the top
        current_node = self.head
        exists = True
        for letter in word:
            if letter in current_node.children:
                current_node = current_node.children[letter]
            else:
                exists = False
                break

        # Still need to check if we just reached a word like 't'
        # that isn't actually a full word in our dictionary
        if exists:
            if current_node.freq == None:
                exists = False

        return exists

    def start_with_prefix(self, prefix):
        """ Returns a list of all words in tree that start with prefix """
        words = list()
        if prefix == None:
            raise ValueError('Requires not-Null prefix')

        # Determine end-of-prefix node
        top_node = self.head
        for letter in prefix:
            if letter in top_node.children:
                top_node = top_node.children[letter]
            else:
                # Prefix not in tree, go no further
                return words

        # Get words under prefix
        if top_node == self.head:
            queue = [node for key, node in top_node.children.iteritems()]
        else:
            queue = [top_node]

        # Perform a breadth first search under the prefix
        # A cool effect of using BFS as opposed to DFS is that BFS will return
        # a list of words ordered by increasing length
        while queue:
            current_node = queue.pop()
            if current_node.data != None:
                # Isn't it nice to not have to go back up the tree?
                words.append(current_node.data)

            queue = [node for key, node in current_node.children.iteritems()] + queue

        return words

    def count(self, word):
        """ This returns the 'data' of the node identified by the given word """
        if not self.has_word(word):
            return 0
        # Race to the bottom, get data
        current_node = self.head
        for letter in word:
            current_node = current_node[letter]

        return current_node.freq

    def total(self):
        return self.sum


class DataLoader(object):
    def __init__(self):
        pass

    def load_data(self, path='data/brown', count=None):
        logging.info("Load data in {}".format(path))
        self.path = path
        self.sentences = []
        self.num_sentences = 0
        self.num_words = 0
        self.num_files = 0
        self.path
        self.count = count
        if os.path.isdir(self.path):
            file_names = os.listdir(self.path)
            for file_name in file_names:
                if re.match('^c.[0-9]+$', file_name):
                    file_path = os.path.join(path, file_name)
                    with open(file_path, 'r') as f:
                        self.num_files += 1
                        for line in f.readlines():
                            tokens = line.lower().strip().split()
                            if len(tokens) > 0:
                                self.sentences.append(np.array([np.array(token.split('/')) for token in tokens]))
                                self.num_sentences += 1
                                self.num_words += len(self.sentences[-1])
                                if self.count is not None and self.num_sentences == self.count:
                                    break

                if self.count is not None and self.num_sentences == self.count:
                    break
        logging.info("Load total {} files :: {} sentences :: {} words".format(self.num_files, self.num_sentences,
                                                                              self.num_words))
        self.sentences = np.squeeze(self.sentences)
        return self.sentences

def merge(ngram):
    return ' '.join(ngram)

if __name__ == '__main__':
    data_loader = DataLoader()
    data_loader.load_data()
