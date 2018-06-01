import numpy as np
import tensorflow as tensorflow
import time
from tensorflow.python.layers.core import Dense
from progressbar import update_progress_bar
import pandas as pd


class KerasTextSummarizer:
    in_verbose_mode = False
    do_print_verbose_header = True
    data = None
    codes = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]
    words_to_ids = {}
    ids_to_words = {}
    embeddings_index = None
    word_counts = {}
    max_word_usage_count = 20
    max_embedding_matrix_size = 300

    sorted_summaries = []
    sorted_texts = []
    max_text_length = 84
    max_summary_length = 13
    min_length = 2
    unk_text_limit = 1
    unk_summary_limit = 0

    def __init__(self, embeddings_index_filename=None, in_verbose_mode=False):
        self.in_verbose_mode = in_verbose_mode
        self.say("In verbose mode")
        self.say("Loading tensorflow and numpy...")
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
        word_vector_info = self.__convert_words_to_vectors()
        summaries_word_vectors = word_vector_info["summaries"]["word_vectors"]
        reviews_word_vectors = word_vector_info["reviews"]["word_vectors"]
        lengths_summaries = self.__create_lengths(summaries_word_vectors)
        lengths_reviews = self.__create_lengths(lengths_summaries)
        self.__sort_summaries()
        self.say("Done loading data")

    def __count_words(self):
        '''Count the number of occurrences of each word in a set of text'''
        self.say("  Counting word occurrences... ", "")
        largest_word_count = 0
        for branch in self.data:
            for row in branch:
                for word in row.split():
                    if word not in self.word_counts:
                        self.word_counts[word] = 1
                    else:
                        self.word_counts[word] += 1
                    if self.word_counts[word] > largest_word_count:
                        largest_word_count = self.word_counts[word]
        self.say("done. Max {:,}.".format(largest_word_count))

    def __build_word_id_table(self):
        self.say("  Creating word vector table...", "")
        self.words_to_ids = {}
        self.ids_to_words = {}
        value = 0
        for word, count in self.word_counts.items():
            if count >= self.max_word_usage_count or \
                    word in self.embeddings_index:
                self.words_to_ids[word] = value
                value += 1
        for code in self.codes:
            self.words_to_ids[code] = len(self.words_to_ids)

        for word, id in self.words_to_ids.items():
            self.ids_to_words[id] = word
        self.say("done. Found {:,} words".format(len(self.ids_to_words)))

    def __build_word_embedding_matrix(self):
        self.say("  Creating word embedding matrix...", "")
        num_words = len(self.words_to_ids)

        # Create matrix with default values of zeroo
        word_embedding_matrix = np.zeros(
            (num_words, self.max_embedding_matrix_size),
            dtype=np.float32
        )
        for word, id in self.words_to_ids.items():
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
        self.say("Done")

    def __convert_words_to_vectors(self):
        '''Convert words in text to an integer.
           If word is not in vocab_to_int, use UNK's integer.
           Total the number of words and UNKs.
           Add EOS token to the end of texts'''
        summaries_num_words = 0
        summaries_num_unknown_words = 0
        reviews_num_words = 0
        reviews_num_unknown_words = 0

        summaries_word_vectors = []
        reviews_word_vectors = []
        self.say("  Loading word vectors... ", "")
        for row in self.data:
            summary_word_vectors = []
            for word in row["summary"].split():
                summaries_num_words += 1
                if word in self.words_to_ids:
                    summary_word_vectors.append(self.words_to_ids[word])
                else:
                    summary_word_vectors.append(self.words_to_ids["<UNK>"])
                    summaries_num_unknown_words += 1
                summaries_word_vectors.append(summary_word_vectors)

            review_word_vectors = []
            for word in row["review"].split():
                reviews_num_words += 1
                if word in self.words_to_ids:
                    review_word_vectors.append(self.words_to_ids[word])
                else:
                    review_word_vectors.append(self.words_to_ids["<UNK>"])
                    reviews_num_unknown_words += 1
                review_word_vectors.append(self.words_to_ids["<EOS>"])
                reviews_word_vectors.append(review_word_vectors)
        result = {
            "summaries": {
                "word_vectors": summaries_word_vectors,
                "num_words": summaries_num_words,
                "num_unknown_words": summaries_num_unknown_words
            },
            "reviews": {
                "word_vectors": reviews_word_vectors,
                "num_words": reviews_num_words,
                "num_unknown_words": reviews_num_unknown_words
            }
        }
        total_words_found = summaries_num_words + reviews_num_words
        total_unknown_words = \
            summaries_num_unknown_words + reviews_num_unknown_words
        percent_unknown_words = total_unknown_words / total_words_found
        self.say(" done. Found {:,} unknown words ({}%).".format(
            total_unknown_words,
            round(100*percent_unknown_words, 2)
        ))
        return result

    def __create_lengths(self, text):
        '''Create a data frame of the sentence lengths from a text'''
        lengths = []
        for sentence in text:
            lengths.append(len(sentence))
        return pd.DataFrame(lengths, columns=['counts'])

    def __sort_summaries(self):
        pass

    def say(self, message, end="\n"):
        if self.in_verbose_mode is True:
            if self.do_print_verbose_header is True:
                print(
                    "[{}]: {}".format(self.__class__.__name__, message),
                    end=end
                )
            else:
                print(message, end=end)
        if end != "\n":
            self.do_print_verbose_header = False
        else:
            self.do_print_verbose_header = True
