#!/usr/bin/env python3
import argparse
import sys
from KerasSummarizer.KerasWordVectorizerManager import \
    KerasWordVectorizerManager


# from tensorflow.ops.rnn_cell_impl import _zero_state_tensors
def _zero_state_tensors(state_size, batch_size, dtype):
    """Create tensors of zeros based on state_size, batch_size, and dtype."""
    def get_state_shape(s):
        """Combine s with batch_size to get a proper tensor shape."""
        c = _concat(batch_size, s)
        size = array_ops.zeros(c, dtype=dtype)
        if not context.executing_eagerly():
            c_static = _concat(batch_size, s, static=True)
            size.set_shape(c_static)
        return size
    return nest.map_structure(get_state_shape, state_size)


def build_command_parser():
    parser = argparse.ArgumentParser(
        description='Retrieves data from a Google Maps search'
    )
    parser.add_argument(
        'embeddings_file',
        help='Load word embeddings file, eg. ConceptNet Numberbach, '
             'GloVe, or Gigaword'
    )
    parser.add_argument(
        'reviews_file',
        help='Load the amazon reviews CSV file, available on kaggle.com'
    )
    parser.add_argument(
        'save_file',
        help='Save the vectorized data into a this file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print debugging messages'
    )

    return parser


def main():
    command_parser = build_command_parser()
    command_arguments = command_parser.parse_args()

    vectorizor_manager = KerasWordVectorizerManager(command_arguments.verbose)

    try:
        sorted_reviews_summaries_word_vectors = \
            vectorizor_manager.build_word_vectors(
                command_arguments.embeddings_file,
                command_arguments.reviews_file
            )
        vectorizor_manager.save_vectors_to_file(
            sorted_reviews_summaries_word_vectors,
            command_arguments.save_file
        )
    except Exception as e:
        sys.exit("Error: {}".format(str(e)))


if __name__ == "__main__":
    main()
