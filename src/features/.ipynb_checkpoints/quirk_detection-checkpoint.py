# src/features/quirk_detection.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def detect_journal_quirks(df, text_col="text_clean", n_clusters=5):
    """
    Clusters journal entries into themes (quirks) using TF-IDF + KMeans.
    Returns the updated DataFrame and fitted model.
    """
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')  # better vectorization
    tfidf_matrix = tfidf.fit_transform(df[text_col])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)
    df['quirk_cluster'] = kmeans.labels_
    return df, kmeans

def summarize_quirks(df, text_col='text_clean', max_samples=3):
    """
    Returns 2-3 sample entries per quirk cluster.
    """
    return df.groupby('quirk_cluster')[text_col].apply(lambda x: list(x)[:max_samples]).to_dict()
