from transformers import pipeline

def get_transformer_sentiment(df):
    sentiment_pipeline = pipeline('sentiment-analysis')
    df['text_transformer'] = df['text'].apply(lambda x: sentiment_pipeline(x)[0]['label'] if x else "")
    #df['title_sentiment_transformer'] = df['cleaned_title'].apply(lambda x: sentiment_pipeline(x)[0]['label'] if x else "")
    return df
