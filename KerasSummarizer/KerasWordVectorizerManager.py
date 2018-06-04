from .KerasWordVectorizer import KerasWordVectorizer
from .DataPreprocessor import DataPreprocessor
import pickle
import gzip
from datetime import datetime


class KerasWordVectorizerManager:
    in_verbose_mode = False
    do_print_verbose_header = True
    summarizer = None
    data_preprocessor = None

    def __init__(
        self,
        in_verbose_mode=False
    ):
        self.in_verbose_mode = in_verbose_mode
        self.say("In verbose mode")
        self.data_preprocessor = DataPreprocessor(
            in_verbose_mode=self.in_verbose_mode
        )

    def get_cleaned_data(
        self,
        reviews_filename
    ):
        self.say("Loading data")
        self.test_file(reviews_filename)
        reviews_summaries = None
        try:
            reviews_summaries = self.data_preprocessor.load_data_from_csv(
                reviews_filename
            )
        except:
            raise Exception("File '{}' was is not CSV".format(
                reviews_filename
            ))
        unwanted_headers = [
            'Id',
            'ProductId',
            'UserId',
            'ProfileName',
            'HelpfulnessNumerator',
            'HelpfulnessDenominator',
            'Score',
            'Time'
        ]
        reviews_summaries = self.data_preprocessor.drop_unwanted_columns(
            reviews_summaries,
            unwanted_headers
        )
        cleaned_reviews_summaries = \
            self.data_preprocessor.clean_reviews_summaries(
                reviews_summaries
            )
        self.say("Done")
        return cleaned_reviews_summaries

    def get_word_vectors(
        self,
        embeddings_filename,
        reviews_summaries
    ):
        self.say("Building word vectors... ")
        self.test_file(embeddings_filename)
        self.summarizer = KerasWordVectorizer(
            embeddings_index_filename=embeddings_filename,
            in_verbose_mode=self.in_verbose_mode
            )
        words_to_vectors, vectors_to_words = \
            self.summarizer.get_vectors_from_data_pairs(
                reviews_summaries
            )
        self.say("Done")
        return words_to_vectors, vectors_to_words

    def get_reviews_summaries_word_vectors(
        self,
        reviews_summaries,
        words_to_vectors
    ):
        self.say("Getting review summary word vectors...")
        sorted_reviews_summaries_word_vectors = \
            self.summarizer.get_reviews_summaries_word_vectors(
                reviews_summaries,
                words_to_vectors
            )
        self.say("Done")
        return sorted_reviews_summaries_word_vectors

    def get_word_embedding_matrix(self, words_to_vectors):
        self.say("Creating word embedding matrix... ")
        word_embedding_matrix = self.summarizer.get_word_embedding_matrix(
            words_to_vectors
        )
        self.say("Done")
        return word_embedding_matrix

    def save_data_to_file(self, data, filename):
        self.say("Saving data to '{}'... ".format(filename), "")
        self.test_file(filename, "wb")
        save_file = gzip.open(filename, 'wb')
        pickle.dump(data, save_file)
        save_file.close()
        self.say("done")

    def test_file(self, filename, mode='r'):
        try:
            with open(filename, mode) as f:
                f.close()
        except:
            access_mode = "readable"
            if 'w' in mode or 'a' in mode:
                access_mode = 'writeable'

            raise Exception("File '{}' was not {}".format(
                filename,
                access_mode
            ))

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
