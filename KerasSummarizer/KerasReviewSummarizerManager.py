from .KerasReviewSummarizer import KerasReviewSummarizer
import pickle
import gzip
from datetime import datetime


class KerasReviewSummarizerManager:
    in_verbose_mode = False
    do_print_verbose_header = True
    keras_summarizer = None
    TEXT_VECTORS_FILE = 'text_vectors.pklz'
    WORDS_TO_VECTORS_FILE = 'word_vectors.pklz'
    WORD_EMBEDDINGS_FILE = 'embeddings.pklz'

    def __init__(
        self,
        in_verbose_mode=False
    ):
        self.in_verbose_mode = in_verbose_mode
        self.say("In verbose mode")

    def load_summarizer(
        self,
        word_vectors,
        words_to_vectors,
        embeddings_index_filename
    ):
        self.say("Loading KerasReviewSummarizer...")
        self.keras_summarizer = KerasReviewSummarizer(
            word_vectors=word_vectors,
            words_to_vectors=words_to_vectors,
            embeddings_index_filename=embeddings_index_filename,
            in_verbose_mode=self.in_verbose_mode
        )
        self.say("Done")

    def build_model(self):
        self.say("Building model...")
        self.keras_summarizer.build_graph()
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
        word_embeddings_filename = "{}{}".format(
            file_prefix,
            self.WORD_EMBEDDINGS_FILE
        )
        self.test_file(words_to_vectors_filename, "rb")
        self.test_file(text_vectors_filename, "rb")
        self.test_file(word_embeddings_filename, "rb")
        file = gzip.open(text_vectors_filename, 'rb')
        text_vectors = pickle.load(file)
        file.close()
        file = gzip.open(words_to_vectors_filename, 'rb')
        words_to_vectors = pickle.load(file)
        file.close()
        file = gzip.open(word_embeddings_filename, 'rb')
        word_embeddings = pickle.load(file)
        file.close()
        self.say("done")
        return text_vectors, words_to_vectors, word_embeddings

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
