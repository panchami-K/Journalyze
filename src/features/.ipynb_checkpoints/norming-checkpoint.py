# src/features/norming.py

from sklearn.cluster import KMeans

def peer_group_clusters(df, feature_cols, n_clusters=3):
    """
    KMeans clustering to group user entries (or users) based on personality traits or derived features.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['peer_group'] = kmeans.fit_predict(df[feature_cols])
    return df, kmeans
