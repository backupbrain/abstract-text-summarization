#!/usr/bin/env python3
import pickle
import gzip


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
        'vectors_file',
        help='Load the vectorized reviews, created with build_word_vectors.py'
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

    try:
        with open(command_arguments.vectors_file, 'r') as f:
            f.close()
    except:
        sys.exit("Error: File '{}' was not readable".format(
            command_arguments.embeddings_file
        ))
    try:
        with open(command_arguments.vectors_file, 'r') as f:
            f.close()
    except:
        sys.exit("Error: File '{}' was not readable".format(
            command_arguments.reviews_file
        ))


if __name__ == "__main__":
    main()
