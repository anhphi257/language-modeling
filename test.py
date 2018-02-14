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
    print a[:-1]
    # print 'yes'
    print a[-1]
if __name__ == '__main__':
    main()
