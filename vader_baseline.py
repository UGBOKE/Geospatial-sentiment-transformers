from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_vader_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    def analyze_sentiment(text):
        if text:
            scores = analyzer.polarity_scores(text)
            compound = scores['compound']
            if compound >= 0.05:
                return 'positive'
            elif compound <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        return 'neutral'
    
    df['vader_sentiment'] = df['text'].apply(analyze_sentiment)
    return df
