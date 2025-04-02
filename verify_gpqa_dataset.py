import pandas as pd

# Load the reformatted dataset
df = pd.read_csv('data/gpqa_diamond.csv')

# Print basic information
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Number of unique IDs: {df['id'].nunique()} (should equal total rows: {len(df)})")

# Check for missing values
missing_values = df.isnull().sum()
print(f"\nMissing values per column:")
for col, count in missing_values.items():
    print(f"  {col}: {count}")

# Display a few samples properly formatted
print("\nSample entries (first 2 rows):")
for i, row in df.head(2).iterrows():
    print(f"\nEntry {i+1}:")
    print(f"  ID: {row['id']}")
    print(f"  Question: {row['question'][:100]}...")  # Show first 100 chars
    print(f"  Solution: {row['solution'][:100]}...")  # Show first 100 chars
    print(f"  Answer: {row['answer']}")

# Verify the answer is contained in the solution
answer_in_solution_count = sum(row['solution'].find(str(row['answer'])) != -1 for _, row in df.iterrows())
print(f"\nEntries where answer is found in solution: {answer_in_solution_count}/{len(df)}")

# Check length statistics
print("\nLength statistics:")
print(f"  Question avg length: {df['question'].str.len().mean():.2f} characters")
print(f"  Solution avg length: {df['solution'].str.len().mean():.2f} characters")
print(f"  Answer avg length: {df['answer'].str.len().mean():.2f} characters") 