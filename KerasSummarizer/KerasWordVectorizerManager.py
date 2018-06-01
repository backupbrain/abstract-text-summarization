from .KerasWordVectorizer import KerasWordVectorizer
from .TextSummaryUtilities import TextSummaryUtilities
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
        summarizer_utilities = TextSummaryUtilities(
            in_verbose_mode=self.in_verbose_mode
        )
        reviews_summaries = None
        try:
            reviews_summaries = summarizer_utilities.load_data_from_csv(
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
        reviews_summaries = summarizer_utilities.drop_unwanted_columns(
            reviews_summaries,
            unwanted_headers
        )
        cleaned_reviews_summaries = \
            summarizer_utilities.clean_reviews_summaries(
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

    def save_vectors_to_file(self, word_vectors, filename):
        self.say("Saving vectors to '{}'... ".format(filename), "")
        self.test_file(filename, "wb")
        save_file = gzip.open(filename, 'wb')
        pickle.dump(word_vectors, save_file)
        save_file.close()
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
