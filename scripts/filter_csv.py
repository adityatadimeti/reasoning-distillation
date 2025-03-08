import pandas as pd

# Read the original CSV file
print("Reading the original CSV file...")
df = pd.read_csv('data/aime/recursive_reasoning_results.csv')

# Print the original dataframe shape
print(f"Original dataframe shape: {df.shape}")

# Filter rows to only include those with non-empty 'new_completion' entries
filtered_df = df[df['new_completion'].notna()]

# Print the filtered dataframe shape
print(f"Filtered dataframe shape: {filtered_df.shape}")

# Save the filtered dataframe to a new CSV file
output_file = 'data/aime/filtered_recursive_reasoning_results.csv'
filtered_df.to_csv(output_file, index=False)
print(f"Filtered data saved to {output_file}")

# Display the IDs of all rows with non-empty new_completion values
print("\nIDs of all rows with non-empty new_completion values:")
print(filtered_df['ID'].tolist())

# Display the first few rows of the filtered dataframe
print("\nFirst few rows of the filtered dataframe:")
print(filtered_df.head())

# Count the actual number of rows in the output file
with open(output_file, 'r', encoding='utf-8') as f:
    line_count = sum(1 for line in f)
print(f"\nNumber of lines in the filtered CSV file: {line_count}")
print(f"Number of data rows (excluding header): {line_count - 1}") 