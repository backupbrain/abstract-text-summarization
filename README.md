# Abstract Text Summarization

Using Keras, Tensorflow, Python, NLTK, and Numberbatch

## Overview

This program learns how to write summaries from Amazon reviews using Deep Learning. It then writes it's own natural language summaries from any new review.

## Setup

### Setup Python

Install python requirements

```
$ pip3 install -r requirements.txt
```

### Get Training Data

This program relies on a specific data set, a set of [~500,000 Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews).


### Get Word Embeddings / Semantic Vector Library

Download the [ConceptNet Numberbatch](https://github.com/commonsense/) semantic vector library. This is necessary for determining what words act as synonyms, etc.

```
$ wget http://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-en-17.02.txt.gz
$ gunzip numberbatch-en-17.02.txt.gz
```

To speed up processing later, create a ramdisk

```
$ sudo mkdir /mnt/numberbatchramdisk
$ sudo mount -t tmpfs -o size=1331M tmpfs /mnt/numberbatchramdisk
$ sudo cp numberbatch-en-17.02.txt /tmp/numberbatchramdisk
```

## Running

The first step is to convert the Amazon reviews data into word vectors. This only needs to be done once each time there is a new data set.

```
$./build_word_vectors.py /path/to/numberbatch-en-17.02.txt /path/to/Reviews.csv /path/to/save_wordvectors.pklz --verbose
```

Example output:

```
[KerasWordVectorizerManager]: In verbose mode
[KerasWordVectorizerManager]: Building word vectors
[TextSummaryUtilities]: In verbose mode
[TextSummaryUtilities]: Loading nltk and pandas...
[TextSummaryUtilities]: Loading CSV file: 'Reviews.csv'... done
[TextSummaryUtilities]: Dropping unwanted columns
[TextSummaryUtilities]:   Original size: (568454, 10)
[TextSummaryUtilities]:   New size: (568411, 2)
[TextSummaryUtilities]: Done
[TextSummaryUtilities]: Cleaning reviews and summaries...
[TextSummaryUtilities]:   Cleaning summaries... done
[TextSummaryUtilities]:   Cleaning reviews... done
[TextSummaryUtilities]: Done
[KerasWordVectorizer]: In verbose mode
[KerasWordVectorizer]: Loading tensorflow and numpy...
[KerasWordVectorizer]: Loading embeddings file 'numberbatch-en-17.02.txt'...done
[KerasWordVectorizer]: Loading data...
[KerasWordVectorizer]:   Counting word occurrences... done. Max 568,410.
[KerasWordVectorizer]:   Creating word vector table...done. Found 6 words
[KerasWordVectorizer]:   Creating word embedding matrix...Done
[KerasWordVectorizer]:   Loading word vectors...  done. Found 48,408,060 unknown words (99.96%).
[KerasWordVectorizer]:   Sorting summaries... 
[KerasWordVectorizer]:     Counting words... done
[KerasWordVectorizer]:    Counting unknown words... done
[KerasWordVectorizer]:    Counting unknown words... done
[KerasWordVectorizer]:   Sorting... done
[KerasWordVectorizer]: Done loading data
[KerasWordVectorizerManager]: Done
[KerasWordVectorizerManager]: Saving vectors to 'save_wordvectors.pklz'... done
```

The next step is to train the machine learning model.

```
$ ./summarizer.py /mnt/numberbatchramdisk/numberbatch-en-17.02.txt wordvectors.pklz --verbose
```