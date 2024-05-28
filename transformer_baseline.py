from transformers import pipeline

def get_transformer_sentiment(df):
    sentiment_pipeline = pipeline('sentiment-analysis')

    def analyze_sentiment(text):
        max_length = 512  # Maximum length for BERT-based models
        chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
        
        results = []
        for chunk in chunks:
            result = sentiment_pipeline(chunk)[0]
            results.append(result['label'])
        
        # Aggregate results (simple majority voting, can be improved)
        positive = results.count('POSITIVE')
        negative = results.count('NEGATIVE')

        if positive > negative:
            return 'POSITIVE'
        elif negative > positive:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
    
    df['transformer_sentiment'] = df['text'].apply(lambda x: analyze_sentiment(x) if x else "NEUTRAL")
    return df
