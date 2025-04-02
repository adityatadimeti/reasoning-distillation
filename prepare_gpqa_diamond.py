import pandas as pd
import uuid

# Download the GPQA Diamond dataset
print("Downloading GPQA Diamond dataset...")
df = pd.read_csv('hf://datasets/Idavidrein/gpqa/gpqa_diamond.csv')
print(f"Downloaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# Prepare the new dataframe with required columns
print("Reformatting data...")
new_data = []

for i, row in enumerate(df.iterrows()):
    # Generate a unique ID that includes the index for easier filtering
    # Format: idx_{index}_{unique_id}
    index = i  # 0-indexed
    unique_part = str(uuid.uuid4())[:8]
    unique_id = f"idx_{index:03d}_{unique_part}"
    
    # Extract the question
    question = row[1]['Question']
    
    # Extract the correct answer
    answer = row[1]['Correct Answer']
    
    # Extract the explanation if available
    explanation = row[1].get('Explanation', '')
    
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