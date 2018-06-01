import pandas as pd
import numpy as numpy
import numpy as numpyimport tensorflow as tensorflowimport re
from nltkpcorpus import stopwords
import time
from tensorflow.python.layers.core import Dense
from nltk_english_contractions import contraction_list


class KerasTextSummarizer:
    data = None
    codes = ["<UNK>","<PAD>","<EOS>","<GO>"]
    word_ids = {}
    id_words = {}
    embeddings_index = None
    word_counts = {}
    max_word_usage_count = 20
    max_embedding_matrix_size = 300

    def __init__(self, embeddings_index_filename=None):
        if embeddings_index_filename is not None:
            self.__load_embeddings_index(embeddings_index_file)

    def __lead_embeddings_index(self, embeddings_index_filename):
        embeddings_index = {}
        with open(, encoding='utf-8') as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = embedding

    def load_data(self, data):
        self.data = data
        self.word_counts = {}
        self.__count_words()
        self.__build_word_id_table()
        self.__build_word_embedding_matrix()

    def __count_words(self):
        '''Count the number of occurrences of each word in a set of text'''
        for branch in self.data:
            for row in branch:
                for word in row.split():
                    if word not in self.word_counts:
                        self.word_counts[word] = 1
                    else:
                        self.word_counts[word] += 1

    def __build_word_id_table(self):
        self.word_ids = {}
        self.id_words = {}
        value = 0
        for word, count in self.word_counts.items():
            if count >= self.max_word_usage_count or word in self.embeddings_index:
                self.word_ids[word] = value
                value += 1
        for code in self.codes:
            self.word_ids[code] = len(self.word_ids)

        for word, id in self.word_ids.items():
            id_words[id] = word

    def __build_word_embedding_matrix(self):
        num_words = len(self.word_ids)
        
        # Create matrix with default values of zeroo
        word_embedding_matrix = np.zeros(
            (nb_words, embedding_dim),
            dtype=np.float32
        )
        for word, id in self.word_ids.items():
            if word in self.embeddings_index:
                word_embedding_matrix[i] = self.embeddings_index[word]
            else:
                # If word not in CN, create a random embedding for it
                new_embedding = np.array(
                    np.random.uniform(-1.0, 1.0, embedding_dim)
                )
                embeddings_index[word] = new_embedding
                word_embedding_matrix[i] = new_embedding
