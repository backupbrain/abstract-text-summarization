#!/usr/bin/env python3
from KerasSummarizer.KerasReviewSummarizerManager import \
    KerasReviewSummarizerManager
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
    word_vectors, words_to_vectors, word_embeddings = \
        summarizer_manager.load_data_from_file(
            command_arguments.load_prefix
        )
    summarizer_manager.load_summarizer(
        word_vectors,
        words_to_vectors,
        command_arguments.embeddings_file
    )
    # except Exception as e:
    #     sys.exit("Error: {}".format(str(e)))


if __name__ == "__main__":
    main()
