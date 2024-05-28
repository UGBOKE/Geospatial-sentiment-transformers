import pandas as pd
import re

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, encoding='latin1')
    cleaning_pattern = r'[^\w\s\']|_|\d|[^\x00-\x7F]+'
    df['cleaned_review'] = df['review'].apply(lambda x: re.sub(cleaning_pattern, '', x) if pd.notnull(x) else "")
    df['cleaned_title'] = df['title'].apply(lambda x: re.sub(cleaning_pattern, '', x) if pd.notnull(x) else "")
    df['text'] = df['cleaned_review'] + " " + df['cleaned_title']
    df['label'] = df['review-label']  # Assuming 'review-label' is the sentiment label
    return df
