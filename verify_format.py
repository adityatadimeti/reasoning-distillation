import pandas as pd

csv_path = "data/gpqa_diamond_mc.csv"
num_rows_to_print = 8

try:
    pd.set_option('display.max_colwidth', None) # Ensure full question text is shown
    df = pd.read_csv(csv_path)
    
    if len(df) < num_rows_to_print:
        print(f"Warning: CSV contains only {len(df)} rows, printing all.")
        num_rows_to_print = len(df)
        
    print(f"--- Displaying 'question' field for first {num_rows_to_print} rows of {csv_path} ---")
    
    for index, row in df.head(num_rows_to_print).iterrows():
        print(f"\n--- ID: {row['id']} (Index: {index}) ---")
        print(row['question'])
        print("-" * 20) # Separator

except FileNotFoundError:
    print(f"Error: File not found at {csv_path}")
except Exception as e:
    print(f"An error occurred: {e}")
