# Abstract Text Summarization

Using Keras, Tensorflow, Python, NLTK, and Numberbatch

## Overview

This program learns how to write summaries from Amazon reviews using Deep Learning. It then writes it's own natural language summaries from any new review.

It is created in an MVC framework so that implementation in other projects is easier.

Inspired by [Currie32's Text-Summarization-with-Amazon-Reviews](https://github.com/Currie32/Text-Summarization-with-Amazon-Reviews/blob/master/summarize_reviews.py)

## Setup

### Setup Python

Install python requirements

```
$ pip3 install -r requirements.txt
```

### Get Training Data

This program relies on a specific data set, a set of [~500,000 Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews).


### Get Word Embeddings / Semantic Vector Library

This project relies on existing trained word-word co-occurrence data. These trained data libraries learn the relationships between words, based on millions of human-written texts. 

### Option 1: ConceptNet Numberbatch

Download the [ConceptNet Numberbatch](https://github.com/commonsense/) semantic vector library. This is necessary for determining what words act as synonyms, etc.

```
$ wget http://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-en-17.02.txt.gz
$ gunzip numberbatch-en-17.02.txt.gz
```

### Option 2: Global Vectors for Word Representation (GloVe)

Download [GloVe](https://nlp.stanford.edu/projects/glove/)

```
$ wget https://nlp.stanford.edu/software/GloVe-1.2.zip
$ unzip GloVe-1.2.zip
```

### Option 3: Gigiword

### Speed tip

These files are large, around 1 Gb. Loading them into a Python script from disk can take 10-20 seconds. They load faster from a ramdisk.

It is possible to move them to a ramdisk like this:

```
$ sudo mkdir /mnt/numberbatchramdisk
$ sudo mount -t tmpfs -o size=1331M tmpfs /mnt/numberbatchramdisk
$ sudo cp numberbatch-en-17.02.txt /tmp/numberbatchramdisk
```

## Running

The first step is to convert the Amazon reviews data into word vectors. This only needs to be done once each time there is a new data set.

```
$./build_word_vectors.py /path/to/numberbatch-en-17.02.txt /path/to/Reviews.csv /path/to/save_text_word_vectors.pklz /path/to/save_word_vector_lookup.pklz --verbose
```

Example output:

```
[KerasWordVectorizerManager]: In verbose mode
[KerasWordVectorizerManager]: Building word vectors
[DataPreprocessor]: In verbose mode
[DataPreprocessor]: Loading nltk and pandas...
[DataPreprocessor]: Loading CSV file: 'Reviews.csv'... done
[DataPreprocessor]: Dropping unwanted columns
[DataPreprocessor]:   Original size: (568454, 10)
[DataPreprocessor]:   New size: (568411, 2)
[DataPreprocessor]: Done
[DataPreprocessor]: Cleaning reviews and summaries...
[DataPreprocessor]:   Cleaning summaries... done
[DataPreprocessor]:   Cleaning reviews... done
[DataPreprocessor]: Done
[KerasWordVectorizer]: In verbose mode
[KerasWordVectorizer]: Loading tensorflow and numpy...
[KerasWordVectorizer]: Loading embeddings file 'numberbatch-en-17.02.txt'... done
[KerasWordVectorizer]: Loading vectors from data pairs...
[KerasWordVectorizer]:   Counting word occurrences... done. Max 568,410.
[KerasWordVectorizer]:   Creating word vector table...done. Found 6 words
[KerasWordVectorizer]:   Creating word embedding matrix...Done
[KerasWordVectorizer]:   Loading word vectors...  done. Found 48,408,060 unknown words (99.96%).
[KerasWordVectorizer]:   Sorting summaries... 
[KerasWordVectorizer]:     Counting words... done
[KerasWordVectorizer]:    Counting unknown words... done
[KerasWordVectorizer]:    Counting unknown words... done
[KerasWordVectorizer]:   Sorting... done
[KerasWordVectorizer]: Done loading vectors
[KerasWordVectorizerManager]: Done
[KerasWordVectorizerManager]: Saving data to 'save_text_word_vectors.pklz'... done
[KerasWordVectorizerManager]: Saving data to 'save_word_vector_lookup.pklz'... done
```

The next step is to train the machine learning model.

```
$ ./summarizer.py /mnt/numberbatchramdisk/numberbatch-en-17.02.txt wordvectors.pklz --verbose
```