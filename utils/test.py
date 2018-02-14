from utils import Trie


def main():
    trie = Trie()
    trie.add("a")
    trie.add("ab")
    trie.add("a")
    trie.add("abc")
    trie.add("a")
    print trie.count("a")

    a = ['a', 'b', 'c']
    a = ['s'] + a
    print a
if __name__ == '__main__':
    main()
