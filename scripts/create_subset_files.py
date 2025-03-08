import pandas as pd

# Read the existing CSV file
input_file = 'data/wrong_seven_deepseek_aime_2024.csv'
print(f"Reading existing CSV file: {input_file}")
df = pd.read_csv(input_file)

# Create the one-row subset file
one_row_df = df.iloc[0:1]
one_row_output = 'data/wrong_one_deepseek_aime_2024.csv'
one_row_df.to_csv(one_row_output, index=False)
print(f"Created one-row CSV file: {one_row_output}")
print(f"Contains ID: {one_row_df['id'].iloc[0]}")

# Create the two-row subset file
two_row_df = df.iloc[0:2]
two_row_output = 'data/wrong_two_deepseek_aime_2024.csv'
two_row_df.to_csv(two_row_output, index=False)
print(f"Created two-row CSV file: {two_row_output}")
print(f"Contains IDs: {two_row_df['id'].tolist()}")

# Print confirmation of completion
print("\nAll files created successfully!")
print(f"1. {one_row_output}: {len(one_row_df)} row")
print(f"2. {two_row_output}: {len(two_row_df)} rows") 