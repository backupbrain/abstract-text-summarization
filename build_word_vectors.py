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
        'save_file_prefix',
        help='Save data into files with this prefix'
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

    #try:
    reviews_summaries = vectorizor_manager.get_cleaned_data(
        command_arguments.reviews_file
    )
    words_to_vectors, vectors_to_words = \
        vectorizor_manager.get_word_vectors(
            command_arguments.embeddings_file,
            reviews_summaries
        )
    word_embedding_matrix = vectorizor_manager.get_word_embedding_matrix(
        words_to_vectors
    )
    sorted_reviews_summaries_word_vectors = \
        vectorizor_manager.get_reviews_summaries_word_vectors(
            reviews_summaries,
            words_to_vectors
        )
    vectorizor_manager.save_data_to_file(
        sorted_reviews_summaries_word_vectors,
        "{}text_vectors.pklz".format(command_arguments.save_file_prefix)
    )
    vectorizor_manager.save_data_to_file(
        words_to_vectors,
        "{}word_vectors.pklz".format(command_arguments.save_file_prefix)
    )
    vectorizor_manager.save_data_to_file(
        words_to_vectors,
        "{}embeddings.pklz".format(command_arguments.save_file_prefix)
    )
    #except Exception as e:
    #    sys.exit("Error: {}".format(str(e)))


if __name__ == "__main__":
    main()
