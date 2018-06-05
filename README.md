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
$./build_word_vectors.py /path/to/numberbatch-en-17.02.txt /path/to/Reviews.csv /path/to/save_prefix_ --verbose
```

Example output:

```
[16:06:11|KerasWordVectorizerManager]: In verbose mode
[16:06:11|DataPreprocessor]: In verbose mode
[16:06:11|DataPreprocessor]: Loading nltk and pandas...
[16:06:11|KerasWordVectorizerManager]: Loading data
[16:06:11|DataPreprocessor]: Loading CSV file: '/path/to/Reviews.csv'... done
[16:06:15|DataPreprocessor]: Dropping unwanted columns
[16:06:15|DataPreprocessor]:   Original size: (568454, 10)
[16:06:16|DataPreprocessor]:   New size: (568411, 2)
[16:06:16|DataPreprocessor]: Done
[16:06:16|DataPreprocessor]: Cleaning reviews and summaries...
[16:06:16|DataPreprocessor]:   Cleaning summaries... done
[16:06:21|DataPreprocessor]:   Cleaning reviews... done
[16:08:41|DataPreprocessor]: Done
[16:08:41|KerasWordVectorizerManager]: Done
[16:08:41|KerasWordVectorizerManager]: Building word vectors... 
[16:08:41|KerasWordVectorizer]: In verbose mode
[16:08:41|KerasWordVectorizer]: Loading tensorflow and numpy...
[16:08:41|KerasWordVectorizer]: Loading embeddings file '/path/to/numberbatch-en-17.02.txt'... done. 484557 word embeddings.
[16:09:17|KerasWordVectorizer]: Loading vectors from data pairs...
[16:09:17|KerasWordVectorizer]:   Counting word occurrences... done. 132,884 words.
[16:09:26|KerasWordVectorizer]:   Found 3,044 unknown words (0%)
[16:09:26|KerasWordVectorizer]:   Creating word vector table... done. Found 65,469 words
[16:09:26|KerasWordVectorizerManager]: Done
[16:09:26|KerasWordVectorizerManager]: Creating word embedding matrix... 
[16:09:26|KerasWordVectorizer]:   Creating word embedding matrix... done. Matrix size is 65,469
[16:09:26|KerasWordVectorizerManager]: Done
[16:09:26|KerasWordVectorizerManager]: Getting review summary word vectors...
[16:09:26|KerasWordVectorizer]: Getting review summary word vectors...
[16:09:26|KerasWordVectorizer]:   Loading word vectors...  done. Found 170,450 unknown words (0.66%).
[16:09:47|KerasWordVectorizer]:   Sorting summaries... 
[16:09:47|KerasWordVectorizer]:     Counting words... done
[16:09:47|KerasWordVectorizer]:     Counting unknown words... done. 0 found
[16:09:49|KerasWordVectorizer]:     Counting unknown words... done. 0 found
[16:09:49|KerasWordVectorizer]:   Sorting... done. 828,529 reviews found
[16:11:49|KerasWordVectorizer]: Done loading vectors
[16:11:50|KerasWordVectorizerManager]: Done
[16:11:50|KerasWordVectorizerManager]: Saving data to '/path/to/save_prefix_text_vectors.pklz'... done
[16:11:57|KerasWordVectorizerManager]: Saving data to '/path/to/save_prefix_word_vectors.pklz'... done
[16:11:57|KerasWordVectorizerManager]: Saving data to '/path/to/save_prefix_embeddings.pklz'... done
```

The next step is to train the machine learning model.

```
$ ./summarizer.py /path/to/numberbatch-en-17.02.txt /path/to/data_prefix --verbose
```
