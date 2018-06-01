from .KerasReviewSummarizer import KerasReviewSummarizer
import pickle
import gzip

class KerasReviewSummarizerManager:
    in_verbose_mode = False
    do_print_verbose_header = True
    keras_summarizer = None

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

    def build_model(self)

    def load_data_from_file(self, filename):
        self.say("Reading data from '{}'... ".format(filename), "")
        self.test_file(filename, "rb")
        file = gzip.open(filename, 'rb')
        data = pickle.load(file)
        data.close()
        return data
        self.say("done")


    def test_file(self, filename, mode='r'):
        try:
            with open(filename, 'r') as f:
                f.close()
        except:
            raise Exception("File '{}' was not readable".format(
                filename
            ))

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
