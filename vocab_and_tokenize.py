import numpy as np
import pandas as pd
import string
import nltk
from collections import defaultdict, Counter

class Vocabulary(object):
    def __init__(self, review_data):
        self.review_data = review_data
        self.reviews_vocab, self.vocab_counts = self.summarize_vocab()
        self.vocab2idx, self.idx2vocab = self.word_index_transform()

    def summarize_vocab(self):
        vocab_counts = Counter()
        reviews_vocab = [] # list to store vocab for all sentences

        for review in self.review_data:
            review_vocab = [] # list to store vocab for the review
            for vocab in nltk.word_tokenize(review):
                if vocab not in string.punctuation:
                    vocab_counts.update([vocab.lower()])
                    review_vocab.append(vocab)
            reviews_vocab.append(review_vocab)

        # Remove the word that only appears once
        vocab_counts = {vocab:occur for vocab, occur in vocab_counts.items() if occur > 1}

        return reviews_vocab, vocab_counts

    def word_index_transform(self):
        # Sorts the words according to the number of appearancesorted
        self.vocab_counts = sorted(self.vocab_counts, key=self.vocab_counts.get, reverse=True)
        self.vocab_counts = ['_PAD','_UNK'] + self.vocab_counts

        vocab2idx = {o:i for i,o in enumerate(self.vocab_counts)}
        idx2vocab = {i:o for i,o in enumerate(self.vocab_counts)}

        return vocab2idx, idx2vocab


class Tokenize(object):
    def __init__(self, review_data, seq_length):
        self.review_data = review_data
        self.seq_length = seq_length

        vocab = Vocabulary(self.review_data)

        self.vocab2idx = vocab.vocab2idx
        reviews_vocab = vocab.reviews_vocab
        self.reivews_vocab_in_index = self.transform_review2index(reviews_vocab)
        self.input_seq = self.padding()

    def transform_review2index(self, reviews_vocab):
        for i, review in enumerate(reviews_vocab):
            reviews_vocab[i] = [self.vocab2idx[vocab] if vocab in self.vocab2idx else 0 for vocab in review]
        reviews_vocab_in_index = reviews_vocab

        return reviews_vocab_in_index

    def padding(self):
        input_seq = np.zeros((len(self.reivews_vocab_in_index), self.seq_length), dtype=int)

        for i, review in enumerate(self.reivews_vocab_in_index):
            input_seq[i, -len(review):] = np.array(review)[:self.seq_length]

        return input_seq
