from transformers import pipeline
import pandas as pd

def load_emotion_model():
    # Using a widely used emotion classifier
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

def load_distortion_model():
    # Placeholder: replace with a custom or fine-tuned model for distortions if available
    # For now, can use same model or a multi-label model trained on distortions
    return pipeline("text-classification", model="j-hartmann/cognitive-distortions-bert", return_all_scores=False)

def add_nlp_emotion_column(df, text_col='text_clean', out_col='nlp_emotion'):
    print("Loading emotion classification model...")
    model = load_emotion_model()
    print("Classifying emotion for each entry (may take several minutes)...")
    df[out_col] = df[text_col].apply(lambda x: model(x)[0]['label'] if x.strip() else "neutral")
    return df

def add_nlp_distortion_column(df, text_col='text_clean', out_col='nlp_distortion'):
    print("Loading distortion classification model...")
    model = load_distortion_model()
    print("Classifying cognitive distortions for each entry (may take several minutes)...")
    df[out_col] = df[text_col].apply(lambda x: model(x)[0]['label'] if x.strip() else "none_detected")
    return df
