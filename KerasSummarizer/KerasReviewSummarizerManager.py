from .KerasReviewSummarizer import KerasReviewSummarizer
from .DataPreprocessor import DataPreprocessor
import pickle
import gzip
from datetime import datetime


class KerasReviewSummarizerManager:
    in_verbose_mode = False
    do_print_verbose_header = True
    keras_summarizer = None
    TEXT_VECTORS_FILE = 'text_vectors.pklz'
    WORDS_TO_VECTORS_FILE = 'word_vectors.pklz'
    VECTORS_TO_WORDS_FILE = 'vectors_to_words.pklz'
    WORD_EMBEDDINGS_FILE = 'embeddings.pklz'

    def __init__(
        self,
        in_verbose_mode=False
    ):
        self.in_verbose_mode = in_verbose_mode
        self.say("In verbose mode")

    def get_cleaned_reviews(self, filename, num_reviews=None):
        data_preprocessor = DataPreprocessor(self.in_verbose_mode)
        reviews_summaries = data_preprocessor.load_data_from_csv(filename)
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
        reviews_summaries = data_preprocessor.drop_unwanted_columns(
            reviews_summaries,
            unwanted_headers
        )
        cleaned_reviews_summaries = \
            data_preprocessor.clean_reviews_summaries(
                reviews_summaries
            )
        return cleaned_reviews_summaries["reviews"]

    def train(self, word_embeddings, word_vectors, words_to_vectors):
        self.keras_summarizer = KerasReviewSummarizer(
            word_embeddings,
            in_verbose_mode=True
        )
        self.say("Training...")
        self.keras_summarizer.train(
            word_vectors["reviews"],
            word_vectors["summaries"],
            words_to_vectors
        )
        self.say("Done")

    def run(
        self,
        word_embeddings,
        word_vectors,
        words_to_vectors,
        vectors_to_words
    ):
        self.keras_summarizer = KerasReviewSummarizer(
            word_embeddings,
            in_verbose_mode=True
        )
        self.say("Running...")
        self.keras_summarizer.run(
            word_vectors,
            words_to_vectors,
            vectors_to_words
        )
        self.say("Done")

    def load_data_from_prefix(self, file_prefix):
        self.say("Loading data files... ", "")
        text_vectors_filename = "{}{}".format(
            file_prefix,
            self.TEXT_VECTORS_FILE
        )
        words_to_vectors_filename = "{}{}".format(
            file_prefix,
            self.WORDS_TO_VECTORS_FILE
        )
        vectors_to_words_filename = "{}{}".format(
            file_prefix,
            self.VECTORS_TO_WORDS_FILE
        )
        word_embeddings_filename = "{}{}".format(
            file_prefix,
            self.WORD_EMBEDDINGS_FILE
        )
        self.test_file(text_vectors_filename, "rb")
        self.test_file(words_to_vectors_filename, "rb")
        self.test_file(vectors_to_words_filename, "rb")
        self.test_file(word_embeddings_filename, "rb")
        file = gzip.open(text_vectors_filename, 'rb')
        text_vectors = pickle.load(file)
        file.close()
        file = gzip.open(words_to_vectors_filename, 'rb')
        words_to_vectors = pickle.load(file)
        file.close()
        file = gzip.open(vectors_to_words_filename, 'rb')
        vectors_to_words = pickle.load(file)
        file.close()
        file = gzip.open(word_embeddings_filename, 'rb')
        word_embeddings = pickle.load(file)
        file.close()
        self.say("done")
        return text_vectors, words_to_vectors, vectors_to_words, word_embeddings

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
