import data_prepration

import train_model

if __name__ == "__main__":

    filepath = r'C:\Users\UGBOKE GEORGE\OneDrive\Documents\TeePublic_review_with_sentiments.csv'

    df = data_prepration.load_and_clean_data(filepath)
        # Step 4: Train a Transformer Model
    model, results = train_model.train_model(df)