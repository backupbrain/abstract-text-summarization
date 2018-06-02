from .KerasWordVectorizer import KerasWordVectorizer
from .DataPreprocessor import DataPreprocessor
import pickle
import gzip


class KerasWordVectorizerManager:
    in_verbose_mode = False
    do_print_verbose_header = True

    def __init__(
        self,
        in_verbose_mode=False
    ):
        self.in_verbose_mode = in_verbose_mode
        self.say("In verbose mode")

    def build_word_vectors(
        self,
        embeddings_filename,
        reviews_filename
    ):
        self.say("Building word vectors... ")
        self.test_file(embeddings_filename)
        self.test_file(reviews_filename)
        data_preprocessor = DataPreprocessor(
            in_verbose_mode=self.in_verbose_mode
        )
        reviews_summaries = None
        try:
            reviews_summaries = data_preprocessor.load_data_from_csv(
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
        reviews_summaries = data_preprocessor.drop_unwanted_columns(
            reviews_summaries,
            unwanted_headers
        )
        cleaned_reviews_summaries = \
            data_preprocessor.clean_reviews_summaries(
                reviews_summaries
            )
        summarizer = KerasWordVectorizer(
            embeddings_index_filename=embeddings_filename,
            in_verbose_mode=self.in_verbose_mode
            )
        words_to_vectors, sorted_reviews_summaries_word_vectors = \
            summarizer.load_vectors_from_data_pairs(cleaned_reviews_summaries)
        self.say("Done")
        return words_to_vectors, sorted_reviews_summaries_word_vectors

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
