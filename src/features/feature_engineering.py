import pandas as pd
from textblob import TextBlob
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# === 1. Text length features ===
def compute_length_features(df, text_col='text_clean'):
    df['text_length'] = df[text_col].apply(len)
    df['word_count'] = df[text_col].apply(lambda x: len(x.split()))
    return df

# === 2. Sentiment features ===
def compute_sentiment(df, text_col='text_clean'):
    df['polarity'] = df[text_col].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df[text_col].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['TextBlob_Analysis'] = df['polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
    return df

# === 3. Cognitive distortion keywords ===
COG_DISTORTION_KEYWORDS = [
    'should', 'always', 'never', 'everyone', 'nobody', 'must', 'can’t', 'nothing'
]

def count_cog_distortions(text):
    tokens = text.lower().split()
    return sum(1 for word in tokens if word in COG_DISTORTION_KEYWORDS)

def compute_cognitive_distortion_score(df, text_col='text_clean'):
    df['cogdist_keyword_count'] = df[text_col].apply(count_cog_distortions)
    return df

# === 4. Negative emotion keywords ===
NEG_EMOTION_WORDS = [
    'anxious', 'worried', 'stressed', 'lonely', 'afraid', 'sad', 'hopeless', 'ashamed', 'angry', 'insecurity', 'resentment', 'guilty'
]

def count_negative_emotions(text):
    tokens = text.lower().split()
    return sum(1 for word in tokens if word in NEG_EMOTION_WORDS)

def compute_emotion_marker_score(df, text_col='text_clean'):
    df['neg_emotion_word_count'] = df[text_col].apply(count_negative_emotions)
    return df

# === 5. TF-IDF features for key psychological terms ===
TFIDF_TERMS = [
    'always', 'never', 'cant', 'im', 'feel', 'everyone', 'nothing', 'must', 'job', 'family', 'relationship',
    'think', 'work', 'career', 'need', 'should'
]

def compute_selected_tfidf_features(df, text_col='text_clean', max_features=100):
    tfidf = TfidfVectorizer(vocabulary=TFIDF_TERMS)
    tfidf_matrix = tfidf.fit_transform(df[text_col])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{w}' for w in tfidf.get_feature_names_out()])
    return pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

# === Main feature pipeline ===
def feature_engineering_pipeline(df, config=None):
    df = compute_length_features(df)
    df = compute_sentiment(df)
    df = compute_cognitive_distortion_score(df)
    df = compute_emotion_marker_score(df)
    df = compute_selected_tfidf_features(df)
    return df
