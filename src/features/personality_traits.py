import pandas as pd

BIG5_LEXICON = {
    'openness': ['imagine', 'creative', 'novel', 'invent', 'art'],
    'conscientiousness': ['organized', 'plan', 'goal', 'order', 'discipline'],
    'extraversion': ['party', 'talk', 'meet', 'group', 'crowd', 'friend'],
    'agreeableness': ['kind', 'forgive', 'generous', 'friendly', 'team'],
    'neuroticism': ['worry', 'anxious', 'sad', 'afraid', 'upset', 'nervous'],
}

def score_big5(text):
    scores = {trait: 0 for trait in BIG5_LEXICON}
    words = text.lower().split()
    for trait, keywords in BIG5_LEXICON.items():
        scores[trait] = sum(w in keywords for w in words)
    return scores

def add_big5_traits(df, text_col="text_clean"):
    big5_scores = df[text_col].apply(score_big5)
    for trait in BIG5_LEXICON:
        df[f'big5_{trait}'] = big5_scores.apply(lambda x: x[trait])
    return df

def get_period_trait_summary(df, trait_cols, date_col='date', freq='M'):
    df[date_col] = pd.to_datetime(df[date_col])
    return df.groupby(pd.Grouper(key=date_col, freq=freq))[trait_cols].mean().reset_index()
