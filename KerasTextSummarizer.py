import numpy as np
import tensorflow as tensorflow
import time
from tensorflow.python.layers.core import Dense


class KerasTextSummarizer:
    in_verbose_mode = False
    do_print_verbose_header = True
    data = None
    codes = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]
    word_ids = {}
    id_words = {}
    embeddings_index = None
    word_counts = {}
    max_word_usage_count = 20
    max_embedding_matrix_size = 300

    def __init__(self, embeddings_index_filename=None, in_verbose_mode=False):
        self.in_verbose_mode = in_verbose_mode
        self.say("In verbose mode")
        if embeddings_index_filename is not None:
            self.__load_embeddings_index(embeddings_index_filename)

    def __load_embeddings_index(self, embeddings_index_filename):
        self.say("Loading embeddings file...", "")
        self.embeddings_index = {}
        with open(embeddings_index_filename, encoding='utf-8') as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = embedding
        self.say("done")

    def load_data(self, data):
        self.say("Loading data...")
        self.data = data
        self.word_counts = {}
        self.__count_words()
        self.__build_word_id_table()
        self.__build_word_embedding_matrix()
        self.say("Done loading data")

    def clean_summaries(self, raw_summaries): 
        summaries = [
            summarizer_utilities.clean_text(summary, remove_stopwords=False)
            for summary in raw_summaries.Summary
        ]
        reviews = [
            summarizer_utilities.clean_text(review, remove_stopwords=False)
            for review in raw_summaries.Text
        ]
        cleaned_review_summaries = []
        for row in range(1, len(summaries)):
            cleaned_review_summaries.append({
                'summary': summaries[row],
                'review': reviews[row]
            })
        return cleaned_review_summaries

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
        self.say("  Creating word vector table...", "")
        self.word_ids = {}
        self.id_words = {}
        value = 0
        for word, count in self.word_counts.items():
            if count >= self.max_word_usage_count or \
                    word in self.embeddings_index:
                self.word_ids[word] = value
                value += 1
        for code in self.codes:
            self.word_ids[code] = len(self.word_ids)

        for word, id in self.word_ids.items():
            self.id_words[id] = word
        self.say("done. Found {} words".format(len(self.id_words)))

    def __build_word_embedding_matrix(self):
        self.say("  Creating word embedding matrix...", "")
        num_words = len(self.word_ids)

        # Create matrix with default values of zeroo
        word_embedding_matrix = np.zeros(
            (num_words, self.max_embedding_matrix_size),
            dtype=np.float32
        )
        for word, id in self.word_ids.items():
            if word in self.embeddings_index:
                word_embedding_matrix[id] = self.embeddings_index[word]
            else:
                # If word not in CN, create a random embedding for it
                new_embedding = np.array(
                    np.random.uniform(
                        -1.0,
                        1.0,
                        self.max_embedding_matrix_size
                    )
                )
                self.embeddings_index[word] = new_embedding
                word_embedding_matrix[id] = new_embedding
        self.say("done")

    def say(self, message, end="\n"):
        if self.in_verbose_mode is True:
            print("[{}]: {}".format(self.__class__.__name__, message), end=end)
        if end != "\n":
            self.do_print_verbose_header = False
        else:
            self.do_print_verbose_header = True


