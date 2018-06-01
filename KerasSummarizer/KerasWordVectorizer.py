import numpy as np


class KerasWordVectorizer:
    in_verbose_mode = False
    do_print_verbose_header = True
    data = None
    codes = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]
    words_to_vectors = {}
    vectors_to_words = {}
    embeddings_index = None
    word_counts = {}
    max_word_usage_count = 20
    max_embedding_matrix_size = 300
    max_review_length = 84
    min_summary_length = 2
    min_review_length = 2
    min_unknown_summary_words = 0
    min_unknown_review_words = 1

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
        self.say(
            "Loading embeddings file '{}'... ".format(
                embeddings_index_filename
            ),
            ""
        )
        self.embeddings_index = {}
        with open(embeddings_index_filename, encoding='utf-8') as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = embedding
        self.say("done")

    def load_vectors_from_data_pairs(self, data):
        self.say("Loading data...")
        self.data = data
        self.word_counts = {}
        self.__count_words()
        self.__build_word_id_table()
        self.__build_word_embedding_matrix()
        word_vector_info = self.__convert_words_to_vectors()
        summaries_word_vectors = word_vector_info["summaries"]["word_vectors"]
        reviews_word_vectors = word_vector_info["reviews"]["word_vectors"]
        sorted_reviews_summaries_word_vectors = self.__sort_summaries(
            summaries_word_vectors,
            reviews_word_vectors
        )
        self.say("Done loading data")
        return sorted_reviews_summaries_word_vectors

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
        self.words_to_vectors = {}
        self.vectors_to_words = {}
        value = 0
        for word, count in self.word_counts.items():
            if count >= self.max_word_usage_count or \
                    word in self.embeddings_index:
                self.words_to_vectors[word] = value
                value += 1
        for code in self.codes:
            self.words_to_vectors[code] = len(self.words_to_vectors)

        for word, id in self.words_to_vectors.items():
            self.vectors_to_words[id] = word
        self.say("done. Found {:,} words".format(len(self.vectors_to_words)))

    def __build_word_embedding_matrix(self):
        self.say("  Creating word embedding matrix...", "")
        num_words = len(self.words_to_vectors)

        # Create matrix with default values of zeroo
        word_embedding_matrix = np.zeros(
            (num_words, self.max_embedding_matrix_size),
            dtype=np.float32
        )
        for word, id in self.words_to_vectors.items():
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
                if word in self.words_to_vectors:
                    summary_word_vectors.append(self.words_to_vectors[word])
                else:
                    summary_word_vectors.append(self.words_to_vectors["<UNK>"])
                    summaries_num_unknown_words += 1
                summaries_word_vectors.append(summary_word_vectors)

            review_word_vectors = []
            for word in row["review"].split():
                reviews_num_words += 1
                if word in self.words_to_vectors:
                    review_word_vectors.append(self.words_to_vectors[word])
                else:
                    review_word_vectors.append(self.words_to_vectors["<UNK>"])
                    reviews_num_unknown_words += 1
                review_word_vectors.append(self.words_to_vectors["<EOS>"])
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

    def __get_text_lengths(self, text):
        self.say("    Counting words... ", "")
        num_words = []
        for row in text:
            num_words.append(len(row))
        self.say("done")
        return num_words

    def __get_num_unknown_words(self, word_vectors):
        '''Counts the number of time UNK appears in a sentence.'''
        self.say("   Counting unknown words... ", "")
        num_unknown_words = 0
        for word in word_vectors:
            if word == self.words_to_vectors["<UNK>"]:
                num_unknown_words += 1
        self.say("done")
        return num_unknown_words

    def __sort_summaries(self, summaries_word_vectors, reviews_word_vectors):
        self.say("  Sorting summaries... ")
        sorted_summary_vectors = []
        sorted_review_vectors = []

        reviews_lengths = self.__get_text_lengths(summaries_word_vectors)
        num_unknown_summary_words = self.__get_num_unknown_words(
            reviews_word_vectors
        )
        num_unknown_review_words = self.__get_num_unknown_words(
            summaries_word_vectors
        )

        self.say("  Sorting... ", "")
        for length in range(
            min(reviews_lengths), self.max_review_length
        ):
            for count, words in enumerate(summaries_word_vectors):
                summary_word_vectors = summaries_word_vectors[count]
                review_word_vectors = reviews_word_vectors[count]
                num_summary_word_vectors = len(summary_word_vectors)
                num_review_word_vectors = len(review_word_vectors)

                if (num_summary_word_vectors >= self.min_summary_length and
                        num_summary_word_vectors <= self.max_summary_length and
                        num_review_word_vectors >= self.min_review_length and
                        num_unknown_summary_words <= self.min_unknown_summary_words and
                        num_unknown_review_words <= self.min_unknown_review_words and
                        length == num_review_word_vectors):
                    sorted_summary_vectors.append(summary_word_vectors)
                    sorted_review_vectors.append(review_word_vectors)

        result = {
            "summaries": sorted_summary_vectors,
            "reviews": sorted_review_vectors
        }
        self.say("done")
        return result

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
