import pandas as pd

# Read the filtered CSV file
input_file = 'data/aime/filtered_recursive_reasoning_results.csv'
print(f"Reading filtered CSV file: {input_file}")
df = pd.read_csv(input_file)

# Extract only the first four columns
subset_df = df[['ID', 'Problem', 'Solution', 'Answer']]

# Rename the columns as specified
subset_df = subset_df.rename(columns={
    'ID': 'id',
    'Problem': 'question',
    'Solution': 'solution',
    'Answer': 'answer'
})

# Save the new dataframe to the specified output file
output_file = 'data/wrong_seven_deepseek_aime_2024.csv'
subset_df.to_csv(output_file, index=False)

# Print information about the created file
print(f"Created new CSV file: {output_file}")
print(f"Number of rows: {len(subset_df)}")
print("\nFirst few rows of the new CSV file:")
print(subset_df.head(3).to_string()) 