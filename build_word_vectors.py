#!/usr/bin/env python3
import argparse
import sys
from KerasSummarizer.KerasWordVectorizerManager import \
    KerasWordVectorizerManager

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
        'vectors_save_file',
        help='Save the vectorized data into a this file'
    )
    parser.add_argument(
        'words_to_vectors_save_file',
        help='Save the words to vectors lookup table into a this file'
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
        words_to_vectors, sorted_reviews_summaries_word_vectors = \
            vectorizor_manager.build_word_vectors(
                command_arguments.embeddings_file,
                command_arguments.reviews_file
            )
        vectorizor_manager.save_data_to_file(
            sorted_reviews_summaries_word_vectors,
            command_arguments.vectors_save_file
        )
        vectorizor_manager.save_data_to_file(
            sorted_reviews_summaries_word_vectors,
            command_arguments.words_to_vectors_save_file
        )
    except Exception as e:
        sys.exit("Error: {}".format(str(e)))


if __name__ == "__main__":
    main()
