#!/usr/bin/env python3
from KerasSummarizer.KerasReviewSummarizerManager import \
    KerasReviewSummarizerManager
from KerasSummarizer.DataPreprocessor import DataPreprocessor
import argparse
import sys


def build_command_parser():
    parser = argparse.ArgumentParser(
        description='Train and write abstract text'
                    'summaries from amazon reviews'
    )
    parser.add_argument(
        'embeddings_file',
        help='Load word embeddings file, eg. ConceptNet Numberbach, '
             'GloVe, or Gigaword'
    )
    parser.add_argument(
        'reviews_file',
        help='Load the Reviews CSV'
    )
    parser.add_argument(
        'load_prefix',
        help='Load the files with this prefix'
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

    summarizer_manager = KerasReviewSummarizerManager(
        command_arguments.verbose
    )

    # try:
    vectors_to_words, words_to_vectors, vectors_to_words, word_embeddings = \
        summarizer_manager.load_data_from_prefix(
            command_arguments.load_prefix
        )
    cleaned_reviews = summarizer_manager.get_cleaned_reviews(
        command_arguments.reviews_file,
        num_reviews=10
    )
    summarizer_manager.run(
        cleaned_reviews,
        vectors_to_words,
        words_to_vectors
    )
    '''
    output_filename = "{}train_data.meta".format(
        command_arguments.load_prefix
    )

    summarizer_manager.train(
        word_vectors["summaries"],
        word_vectors["reviews"],
        train_graph,
        model,
        words_to_vectors,
        output_filename
    )
    '''
    # except Exception as e:
    #     sys.exit("Error: {}".format(str(e)))


if __name__ == "__main__":
    main()
