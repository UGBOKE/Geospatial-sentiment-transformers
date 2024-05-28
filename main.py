import data_prepration
import vader_baseline
import transformer_baseline
import train_model


if __name__ == "__main__":
    # Step 1: Load and clean data
    filepath = r'C:\Users\UGBOKE GEORGE\OneDrive\Documents\TeePublic_review.csv'

    df = data_prepration.load_and_clean_data(filepath)

     # Step 2: VADER Baseline
    df = vader_baseline.get_vader_sentiment(df)
    print("VADER Sentiment Analysis Results:")
    print(df[['text', 'vader_sentiment']].head())
    
    # Step 3: Pretrained Transformer Model without Fine-Tuning
    df = transformer_baseline.get_transformer_sentiment(df)
    print("Transformer Sentiment Analysis Results:")
    print(df[['text', 'transformer_sentiment']].head())
    
    df.to_csv(r'C:\Users\UGBOKE GEORGE\OneDrive\Documents\TeePublic_review_with_sentiments.csv', index=False)

    # Step 4: Train a Transformer Model
    #model, results = train_model.train_model(df)
