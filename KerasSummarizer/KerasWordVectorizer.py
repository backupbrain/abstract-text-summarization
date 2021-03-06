import numpy as np
from datetime import datetime


class KerasWordVectorizer:
    in_verbose_mode = False
    do_print_verbose_header = True
    TEXT_CODES = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]
    MAX_WORD_USAGE_COUNT = 20
    MAX_EMBEDDING_MATRIX_SIZE = 300
    MAX_REVIEW_LENGTH = 84
    MIN_SUMMARY_LENGTH = 2
    MAX_SUMMARY_LENGTH = 13
    MIN_REVIEW_LENGTH = 2
    MIN_UNKNOWN_SUMMARY_WORDS = 0
    MIN_UNKNOWN_REVIEW_WORDS = 1
    embeddings_index = None

    def __init__(self, embeddings_index_filename, in_verbose_mode=False):
        self.in_verbose_mode = in_verbose_mode
        self.say("In verbose mode")
        self.say("Loading tensorflow and numpy...")
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
        self.say("done. {} word embeddings.".format(
            len(self.embeddings_index))
        )

    def get_vectors_from_data_pairs(self, reviews_summaries):
        self.say("Loading vectors from data pairs...")
        reviews_summaries_word_counts = self.__count_words(reviews_summaries)
        self.__count_unknown_words(reviews_summaries_word_counts)
        words_to_vectors, vectors_to_words = \
            self.__build_word_vector_table(reviews_summaries_word_counts)
        return words_to_vectors, vectors_to_words

    def get_reviews_summaries_word_vectors(
        self,
        reviews_summaries,
        words_to_vectors
    ):
        self.say("Getting review summary word vectors...")
        word_vector_info = self.__convert_words_to_vectors(
            words_to_vectors,
            reviews_summaries
        )
        summaries_word_vectors = word_vector_info["summaries"]["word_vectors"]
        reviews_word_vectors = word_vector_info["reviews"]["word_vectors"]
        sorted_reviews_summaries_word_vectors = self.__sort_summaries(
            summaries_word_vectors,
            reviews_word_vectors,
            words_to_vectors
        )
        self.say("Done loading vectors")
        return sorted_reviews_summaries_word_vectors

    def __count_unknown_words(self, word_counts):
        num_unknown_words = 0
        for word, count in word_counts.items():
            if count > self.MAX_WORD_USAGE_COUNT:
                if word not in self.embeddings_index:
                    num_unknown_words += 1
                    
        percent_unknown_words = round(
            num_unknown_words / len(word_counts)
        ) * 100
        self.say("  Found {:,} unknown words ({}%)".format(
            num_unknown_words,
            percent_unknown_words
        ))
        return num_unknown_words

    def __count_words(self, reviews_summaries):
        '''Count the number of occurrences of each word in a set of text'''
        self.say("  Counting word occurrences... ", "")
        word_counts = {}
        for row in reviews_summaries:
            for branch, text in row.items():
                for word in text.split():
                    if word not in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
        self.say("done. {:,} words.".format(len(word_counts)))
        return word_counts

    def __build_word_vector_table(self, reviews_summaries_word_counts):
        self.say("  Creating word vector table... ", "")
        words_to_vectors = {}
        vectors_to_words = {}
        vector = 0
        for word, count in reviews_summaries_word_counts.items():
            if count >= self.MAX_WORD_USAGE_COUNT or \
                    word in self.embeddings_index:
                words_to_vectors[word] = vector
                vector += 1
        for code in self.TEXT_CODES:
            words_to_vectors[code] = len(words_to_vectors)

        for word, vector in words_to_vectors.items():
            vectors_to_words[vector] = word
        self.say("done. Found {:,} words".format(len(vectors_to_words)))
        return words_to_vectors, vectors_to_words

    def get_word_embedding_matrix(self, words_to_vectors):
        self.say("  Creating word embedding matrix... ", "")
        num_words = len(words_to_vectors)

        # Create matrix with default values of zeroo
        word_embedding_matrix = np.zeros(
            (num_words, self.MAX_EMBEDDING_MATRIX_SIZE),
            dtype=np.float32
        )
        for word, id in words_to_vectors.items():
            if word in self.embeddings_index:
                word_embedding_matrix[id] = self.embeddings_index[word]
            else:
                # If word not in CN, create a random embedding for it
                new_embedding = np.array(
                    np.random.uniform(
                        -1.0,
                        1.0,
                        self.MAX_EMBEDDING_MATRIX_SIZE
                    )
                )
                self.embeddings_index[word] = new_embedding
                word_embedding_matrix[id] = new_embedding
        self.say("done. Matrix size is {:,}".format(
            len(word_embedding_matrix))
        )
        return word_embedding_matrix

    def __convert_words_to_vectors(self, words_to_vectors, reviews_summaries):
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
        for row in reviews_summaries:
            summary_word_vectors = []
            for word in row["summary"].split():
                summaries_num_words += 1
                if word in words_to_vectors:
                    summary_word_vectors.append(words_to_vectors[word])
                else:
                    summary_word_vectors.append(words_to_vectors["<UNK>"])
                    summaries_num_unknown_words += 1
                summaries_word_vectors.append(summary_word_vectors)

            review_word_vectors = []
            for word in row["review"].split():
                reviews_num_words += 1
                if word in words_to_vectors:
                    review_word_vectors.append(words_to_vectors[word])
                else:
                    review_word_vectors.append(words_to_vectors["<UNK>"])
                    reviews_num_unknown_words += 1
                review_word_vectors.append(words_to_vectors["<EOS>"])
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

    def __get_num_unknown_words(self, word_vectors, words_to_vectors):
        '''Counts the number of time UNK appears in a sentence.'''
        self.say("    Counting unknown words... ", "")
        num_unknown_words = 0
        for word in word_vectors:
            if word == words_to_vectors["<UNK>"]:
                num_unknown_words += 1
        self.say("done. {:,} found".format(num_unknown_words))
        return num_unknown_words

    def __sort_summaries(self,
                         summaries_word_vectors,
                         reviews_word_vectors,
                         words_to_vectors
                         ):
        self.say("  Sorting summaries... ")
        sorted_summary_vectors = []
        sorted_review_vectors = []

        reviews_lengths = self.__get_text_lengths(summaries_word_vectors)
        num_unknown_summary_words = self.__get_num_unknown_words(
            reviews_word_vectors,
            words_to_vectors
        )
        num_unknown_review_words = self.__get_num_unknown_words(
            summaries_word_vectors,
            words_to_vectors
        )

        self.say("  Sorting... ", "")
        for length in range(
            min(reviews_lengths), self.MAX_REVIEW_LENGTH
        ):
            for vector, words in enumerate(summaries_word_vectors):
                summary_word_vectors = summaries_word_vectors[vector]
                review_word_vectors = reviews_word_vectors[vector]
                num_summary_word_vectors = len(summary_word_vectors)
                num_review_word_vectors = len(review_word_vectors)

                if (num_summary_word_vectors >= self.MIN_SUMMARY_LENGTH and
                        num_summary_word_vectors <= self.MAX_SUMMARY_LENGTH and
                        num_review_word_vectors >= self.MIN_REVIEW_LENGTH and
                        num_unknown_summary_words <=
                    self.MIN_UNKNOWN_SUMMARY_WORDS and
                        num_unknown_review_words <=
                    self.MIN_UNKNOWN_REVIEW_WORDS and
                        length == num_review_word_vectors):
                    sorted_summary_vectors.append(summary_word_vectors)
                    sorted_review_vectors.append(review_word_vectors)

        result = {
            "summaries": sorted_summary_vectors,
            "reviews": sorted_review_vectors
        }
        self.say("done. {:,} reviews found".format(len(result["reviews"])))
        return result

    def say(self, message, end="\n"):
        if self.in_verbose_mode is True:
            if self.do_print_verbose_header is True:
                current_time = datetime.now().strftime('%H:%M:%S')
                print(
                    "[{}|{}]: {}".format(
                        current_time,
                        self.__class__.__name__,
                        message
                    ),
                    end=end
                )
            else:
                print(message, end=end)
        if end != "\n":
            self.do_print_verbose_header = False
        else:
            self.do_print_verbose_header = True
