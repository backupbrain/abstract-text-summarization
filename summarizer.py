#!/usr/bin/env python3
from KerasSummarizer.KerasReviewSummarizer import KerasReviewSummarizer

def build_command_parser():
    parser = argparse.ArgumentParser(
        description='Train and write abstract text summaries from amazon reviews'
    )
    parser.add_argument(
        'embeddings_file',
        help='Load word embeddings file, eg. ConceptNet Numberbach, '
             'GloVe, or Gigaword'
    )
    parser.add_argument(
        'word_vectors_file',
        help='Load the vectorized reviews, created with build_word_vectors.py'
    )
    parser.add_argument(
        'words_to_vectors_file',
        help='Load the word/vector lookup data, created with build_word_vectors.py'
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

    try:
        word_vectors = summarizer_manager.load_data_from_file(
            command_arguments.word_vectors_file
        )
        words_to_vectors = summarizer_manager.load_data_from_file(
            command_arguments.words_to_vectors_file
        )
        summarizer_manager.save_vectors_to_file(
            sorted_reviews_summaries_word_vectors,
            command_arguments.save_file
        )
    except Exception as e:
        sys.exit("Error: {}".format(str(e)))


if __name__ == "__main__":
    main()
