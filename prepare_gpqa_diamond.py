import pandas as pd
import uuid

# Download the GPQA Diamond dataset
print("Downloading GPQA Diamond dataset...")
df = pd.read_csv('hf://datasets/Idavidrein/gpqa/gpqa_diamond.csv')
print(f"Downloaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# Prepare the new dataframe with required columns
print("Reformatting data...")
new_data = []

for _, row in df.iterrows():
    # Generate a unique ID
    unique_id = str(uuid.uuid4())[:8]
    
    # Extract the question
    question = row['Question']
    
    # Extract the correct answer
    answer = row['Correct Answer']
    
    # Extract the explanation if available
    explanation = row.get('Explanation', '')
    
    # Combine explanation and answer into solution
    solution = f"{explanation}\n\nThe answer is: {answer}"
    
    new_data.append({
        'id': unique_id,
        'question': question,
        'solution': solution,
        'answer': answer
    })

# Create new dataframe with the required structure
new_df = pd.DataFrame(new_data)

# Save to CSV
output_path = 'data/gpqa_diamond.csv'
new_df.to_csv(output_path, index=False)
print(f"Saved reformatted dataset to {output_path}")

# Print preview
print("\nPreview of the reformatted dataset:")
print(new_df.head(2))
print(f"Total samples: {new_df.shape[0]}") 