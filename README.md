# Abstract Text Summarization

Using Keras, Tensorflow, Python, NLTK, and Numberbach

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

Download the [ConceptNet Numberbach](https://github.com/commonsense/) semantic vector library. This is necessary for determining what words act as synonyms, etc.

```
$ wget http://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-en-17.02.txt.gz
$ gunzip numberbach-en-17.02.txt.gz
```

To speed up processing later, create a ramdisk

```
$ sudo mkdir /mnt/numberbachramdisk
$ sudo mount -t tmpfs -o size=1331M tmpfs /mnt/numberbachramdisk
$ sudo cp numberbach-en-17.02.txt /tmp/numberbachramdisk
```

## Running

The first step is to convert the Amazon reviews data into word vectors. This only needs to be done once each time there is a new data set.

```
$./build_word_vectors.py Reviews.csv wordvectors.pklz --verbose
```

The next step is to train the machine learning model.

```
$ ./summarizer.py /mnt/numberbachramdisk/numberbach-en-17.02.txt wordvectors.pklz --verbose
```