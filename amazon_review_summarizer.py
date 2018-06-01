#!/usr/bin/env python3
import argparse
import sys
from KerasTextSummarizer import KerasTextSummarizer
from TextSummaryUtilities import TextSummaryUtilities

# Embeddings Index
embeddings_index_filename = 'numberbatch-en-17.02.txt'


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
        'reviews_file',
        help='Load the amazon reviews CSV file, available on kaggle.com')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print debugging messages')

    return parser


def main():
    command_parser = build_command_parser()
    command_arguments = command_parser.parse_args()

    try:
        with open(command_arguments.reviews_file, 'r') as f:
            f.close()
    except:
        sys.exit("Error: File '{}' was not readable".format(
            command_arguments.reviews_file
        ))

    summarizer_utilities = TextSummaryUtilities(
        in_verbose_mode=command_arguments.verbose
    )
    reviews_summaries = None
    #try:
    reviews_summaries = summarizer_utilities.load_data_from_csv(
        command_arguments.reviews_file
    )
    #except:
    #    sys.exit("Error: File '{}' was is not a valid CSV file".format(
    #        command_arguments.reviews_file
    #    ))

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
    summaries = [
        summarizer_utilities.clean_text(summary, remove_stopwords=False)
        for summary in reviews_summaries.Summary
    ]
    reviews = [
        summarizer_utilities.clean_text(review, remove_stopwords=False)
        for review in reviews_summaries.Text
    ]
    cleaned_review_summaries = []
    for row in range(1, len(summaries)):
        cleaned_review_summaries.append({
            'summary': summaries[row],
            'review': reviews[row]
        })

    summarizer = KerasTextSummarizer(
        embeddings_index_filename=embeddings_index_filename,
        in_verbose_mode=command_arguments.verbose
    )
    summarizer.load_data(cleaned_review_summaries)


if __name__ == "__main__":
    main()
