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
    
    df['review_sentiment_vader'] = df['cleaned_review'].apply(analyze_sentiment)
    df['title_sentiment_vader'] = df['cleaned_title'].apply(analyze_sentiment)
    return df
