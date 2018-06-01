import tensorflow as tensorflow
import time
from tensorflow.python.layers.core import Dense
import numpy as np


class KerasReviewSummarizer:
    in_verbose_mode = False
    do_print_verbose_header = True
    embeddings_index = None

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
