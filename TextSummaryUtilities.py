import pandas as pd
import re
from nltk.corpus import stopwords
from nltk_english_contractions import contraction_list


class TextSummaryUtilities:

    def load_data_from_csv(self, filename):
        data = pd.read_csv(filename)
        return data

    def drop_unwanted_columns(self, data, headers=None):
        data = data.dropna()
        if isinstance(headers, type([])):
            data = data.drop(headers, 1)
        return data

    def clean_text(self, text, remove_stopwords=True):
        '''
        Remove unwanted characters, stopwords,
        and format the text to create fewer nulls word embeddings
        '''
        # Convert words to lower case
        text = text.lower()

        # Replace contractions with their longer forms
        if True:
            text = text.split()
            expanded_text = []
            for word in text:
                if word in contraction_list:
                    expanded_text.append(contraction_list[word])
                else:
                    expanded_text.append(word)
            text = " ".join(expanded_text)

        # Format words and remove unwanted characters
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text)
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'\'', ' ', text)

        # Optionally, remove stop words
        if remove_stopwords:
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if w not in stops]
            text = " ".join(text)

        return text

    def clean_summaries(self, data):
        # Clean the summaries and texts
        clean_summaries = []
        for summary in data.Summary:
            clean_summaries.append(
                self.clean_text(summary, remove_stopwords=False)
            )
        print("Summaries are complete.")

    def clean_source_text(self, data):
        clean_texts = []
        for text in data.Text:
            clean_texts.append(self.clean_text(text))
        print("Texts are complete.")
