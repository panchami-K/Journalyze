# src/insights/psychological_inference.py

import pandas as pd

# Example mapping dictionaries (expand as needed)
CBT_DISTORTION_KEYWORDS = {
    'catastrophizing': ['never', 'always', 'worst', 'disaster', 'ruined'],
    'fortune telling': ['will', 'predict', 'inevitable'],
    'personalization': ['my fault', "it's me", 'because of me'],
    'should statements': ['should', 'must', 'have to'],
    'mind reading': ['think', 'assume', 'guess', 'know what others'],
    'overgeneralization': ['everyone', 'nobody', 'every time', 'all', 'none'],
}

# Helper: Given a text, map keywords to distortion types
def infer_distortions(text):
    detected = set()
    text_lower = text.lower()
    for dist_type, keywords in CBT_DISTORTION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            detected.add(dist_type)
    return list(detected) if detected else ["none detected"]

# Add new insight columns to each entry
def psychological_inference(df, text_col='text_clean'):
    df['detected_distortions'] = df[text_col].apply(infer_distortions)
    df['distortion_count'] = df['detected_distortions'].apply(lambda x: 0 if x == ["none detected"] else len(x))
    return df

# Generate a quick summary of detected psychological loops
def user_insight_report(df):
    summary = {}

    # Frequency of each detected distortion
    all_dists = sum(df['detected_distortions'], [])  # Flatten list
    dist_counter = pd.Series(all_dists).value_counts()
    summary['distortion_frequencies'] = dist_counter.to_dict()

    # Feedback examples (expand or personalize as needed)
    summary['feedback'] = {
        'catastrophizing': "Try evidence-based thinking: What is the most likely outcome?",
        'fortune telling': "Notice prediction-based thoughts and test their truth in reality.",
        'personalization': "Ask: Are there other factors at play, not just you?",
        'should statements': "Challenge rigid 'shoulds'—are they preferences or necessities?",
        'mind reading': "Check: Do you have evidence for what others think?",
        'overgeneralization': "Consider: Is this always true, or just sometimes?"
    }

    # Example: Common cyclical language triggers
    summary['common_triggers'] = df['text_clean'].str.extractall(r'\b(never|always|everybody|nobody|should|must)\b')[0].value_counts().to_dict()
    
    return summary

def print_user_insight_report(summary):
    print("\n--- PSYCHOLOGICAL INSIGHT REPORT ---")
    print("Most Frequent Detected Distortions:")
    for dist, count in summary['distortion_frequencies'].items():
        print(f" - {dist}: {count}")
    print("\nCommon Triggers in Language:")
    for trig, count in summary['common_triggers'].items():
        print(f" - {trig}: {count}")
    print("\nSample Feedback Suggestions:")
    for dist, fb in summary['feedback'].items():
        print(f" - {dist}: {fb}")
