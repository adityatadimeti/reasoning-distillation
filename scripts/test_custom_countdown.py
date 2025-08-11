#!/usr/bin/env python3
"""
Test the custom countdown generation with a small sample.
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent))
from generate_custom_countdown import CountdownProblemGenerator

def test_generation():
    """Test generation with small samples."""
    generator = CountdownProblemGenerator()
    
    print("Testing countdown problem generation...")
    print("=" * 50)
    
    # Test each difficulty level with 5 problems
    for num_count in range(4, 11):
        print(f"\n--- Testing {num_count}-number problems ---")
        problems = generator.generate_problems(num_count, count=5)
        
        for problem in problems:
            print(f"\nID: {problem['id']}")
            print(f"Target: {problem['answer']}")
            print(f"Solution: {problem['solution']}")
            
            # Extract numbers from question
            import re
            match = re.search(r"Using the numbers ([\d, ]+), create", problem['question'])
            if match:
                nums_str = match.group(1)
                numbers = [int(n.strip()) for n in nums_str.split(',')]
                print(f"Numbers: {numbers} ({len(numbers)} total)")
                
                # Verify all numbers used
                solution_numbers = [int(n) for n in re.findall(r'\d+', problem['solution'])]
                if sorted(numbers) == sorted(solution_numbers):
                    print("✓ All numbers used exactly once")
                else:
                    print("✗ Number usage mismatch!")
                    print(f"  Expected: {sorted(numbers)}")
                    print(f"  Found: {sorted(solution_numbers)}")
    
    print("\n" + "=" * 50)
    print("Test complete!")

if __name__ == "__main__":
    test_generation()