import pandas as pd
import re

df = pd.read_csv(r'C:\Users\UGBOKE GEORGE\OneDrive\Documents\TeePublic_review.csv', encoding='latin1')

print("Original Data:")
print(df.head())

cleaning_pattern = r'[^\w\s\']|_|\d|[^\x00-\x7F]+'

df['review'] = df['review'].apply(lambda x: re.sub(cleaning_pattern, '', x) if pd.notnull(x) else "")

df['title'] = df['title'].apply(lambda x: re.sub(cleaning_pattern, '', x) if pd.notnull(x) else "")


df['text'] = df['review'] + df['title']