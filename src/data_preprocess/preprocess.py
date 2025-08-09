# src/data_preprocess/preprocess.py
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import yaml

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def load_yaml_config(path='src/config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_text(text, config):
    # Lowercase
    if config['preprocessing'].get('lowercase', True):
        text = text.lower()

    # Remove punctuation
    if config['preprocessing'].get('remove_punctuation', True):
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    if config['preprocessing'].get('remove_numbers', True):
        text = re.sub(r'\d+', '', text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    if config['preprocessing'].get('remove_stopwords', True):
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    if config['preprocessing'].get('lemmatize', True):
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)

def preprocess_dataframe(df, config):
    text_columns = config['preprocessing'].get('text_columns', [])
    for col in text_columns:
        if col in df.columns:
            df[f"{col}_clean"] = df[col].astype(str).map(lambda x: preprocess_text(x, config))
    return df
