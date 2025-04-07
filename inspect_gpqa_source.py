from datasets import load_dataset

def inspect_source_data(indices):
    print(f"Inspecting raw data for indices: {indices}")
    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        print("Dataset loaded.")
        for i in indices:
            if i < len(ds):
                example = ds[i]
                print(f"\n--- Index {i} ---")
                print(f"Question: {repr(example.get('Question'))}")
                print(f"Correct Answer: {repr(example.get('Correct Answer'))}")
                print(f"Incorrect Answer 1: {repr(example.get('Incorrect Answer 1'))}")
                print(f"Incorrect Answer 2: {repr(example.get('Incorrect Answer 2'))}")
                print(f"Incorrect Answer 3: {repr(example.get('Incorrect Answer 3'))}")
            else:
                print(f"Index {i} is out of bounds.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    inspect_source_data([0, 4])
