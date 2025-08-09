import pandas as pd
from collections import Counter

def get_top_ngrams(texts, n=2, top_k=15):
    """Return top K ngrams (words/phrases) in the dataset."""
    all_tokens = []
    for txt in texts:
        words = txt.split()
        ngrams = zip(*[words[i:] for i in range(n)])
        ngram_tokens = [' '.join(ngram) for ngram in ngrams]
        all_tokens.extend(ngram_tokens)
    return Counter(all_tokens).most_common(top_k)

def column_recurrence(df, column, top_n=10):
    """Return the most frequent (recurring) values in a given column."""
    return df[column].value_counts().head(top_n).to_dict()

def detect_recurring_patterns(df, config):
    report = {}

    # Recurring n-grams in text_clean (bigrams, trigrams)
    report['top_bigrams'] = get_top_ngrams(df['text_clean'], n=2, top_k=15)
    report['top_trigrams'] = get_top_ngrams(df['text_clean'], n=3, top_k=15)

    # Recurring emotions & biases
    if 'emotion' in df.columns:
        report['top_emotions'] = column_recurrence(df, 'emotion', top_n=10)
    if 'bias/distortion' in df.columns:
        report['top_biases'] = column_recurrence(df, 'bias/distortion', top_n=10)

    # Recurring triggers (words to watch: "never", "should", "always", "everyone", etc.)
    trigger_terms = ['never', 'should', 'always', 'everyone', 'must', 'can’t', 'nothing', 'everybody']
    trigger_counts = {term: df['text_clean'].str.contains(term).sum() for term in trigger_terms}
    report['trigger_word_counts'] = trigger_counts

    # Recurring scenarios (from 'context', if present)
    if 'context' in df.columns:
        report['top_context'] = column_recurrence(df, 'context', top_n=7)

    return report

def print_pattern_report(report):
    print("\n--- PATTERN & TRIGGER REPORT ---")
    print("Top Bigrams:", report['top_bigrams'])
    print("Top Trigrams:", report['top_trigrams'])
    print("Top Emotions:", report.get('top_emotions'))
    print("Top Biases:", report.get('top_biases'))
    print("Top Contexts:", report.get('top_context'))
    print("Trigger Word Frequency:", report['trigger_word_counts'])
