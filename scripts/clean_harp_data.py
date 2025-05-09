import pandas as pd

# Read the CSV file
df = pd.read_csv('data/harp_level6.csv')

# Create a copy of the dataframe
df_clean = df.copy()

# Clean the answer column by removing '$' symbols
df_clean['answer'] = df_clean['answer'].apply(lambda x: str(x).replace('$', ''))

# Save the cleaned data to a new CSV file
df_clean.to_csv('data/harp_level6_clean.csv', index=False)

print("Cleaned data saved to 'data/harp_level6_clean.csv'") 