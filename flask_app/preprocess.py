import pandas as pd
import numpy as np
import re
import nltk
import sys
import logging
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('wordnet')
nltk.download('stopwords')


def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)


def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text


def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)


def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()


def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def normalize_text(text):
    text = lower_case(text)
 
    text = remove_stop_words(text)

    text = removing_numbers(text)
    
    text = removing_punctuations(text)
    
    text = removing_urls(text)
 
    text = lemmatization(text)
    
    return text
