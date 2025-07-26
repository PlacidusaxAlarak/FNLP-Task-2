import pickle
from collections import Counter

class Vocabulary:
    def __init__(self, pad_token='<pad>', unk_token='<unk>'):
        self.pad_token=pad_token
        self.unk_token=unk_token
        self.word2idx={self.pad_token:0, self.unk_token:1}
        self.idx2word={0:self.pad_token, 1:self.unk_token}

    def add_word(self, word):
        if word not in self.word2idx:
            idx=len(self.word2idx)
            self.word2idx[word]=idx
            self.idx2word[idx]=word

    def __len__(self):
        return len(self.word2idx)

    @classmethod#表示是类方法而不是实例方法
    def build_from_sentences(cls, sentences, min_freq=1):#cls表示类本身
        vocab=cls()
        word_counts=Counter(word for sent in sentences for word in sent)
        for word, freq in word_counts.items():
            if freq>=min_freq:
                vocab.add_word(word)
        return vocab


    def save(self, path):
        #将词汇表对象保存到文件
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=4)

    @staticmethod
    def load(path):
        #从文件中加载词汇表对象
        with open(path, 'rb') as f:
            return pickle.load(f)
