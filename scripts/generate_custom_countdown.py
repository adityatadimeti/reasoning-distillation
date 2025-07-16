#!/usr/bin/env python3
"""
Generate custom countdown problems with varying difficulty levels.
Difficulty is controlled by the number of input numbers (4-10).
"""

import random
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Set
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.countdown_solver import solve_countdown_puzzle

class CountdownProblemGenerator:
    def __init__(self, number_range=(1, 200), target_range=(1, 100)):
        self.number_range = number_range
        self.target_range = target_range
        self.generated_problems = []
        
    def generate_random_numbers(self, count: int) -> List[int]:
        """Generate unique random numbers."""
        # Ensure good distribution across the range
        numbers = []
        
        # Divide range into segments for better distribution
        segment_size = (self.number_range[1] - self.number_range[0]) // count
        
        for i in range(count):
            # Add some randomness to segment selection
            if random.random() < 0.7:  # 70% chance to pick from designated segment
                start = self.number_range[0] + i * segment_size
                end = min(start + segment_size, self.number_range[1])
            else:  # 30% chance to pick from entire range
                start = self.number_range[0]
                end = self.number_range[1]
            
            # Generate unique number
            attempts = 0
            while attempts < 100:
                num = random.randint(start, end)
                if num not in numbers:
                    numbers.append(num)
                    break
                attempts += 1
        
        # Shuffle to avoid predictable ordering
        random.shuffle(numbers)
        return numbers
    
    def find_solvable_target(self, numbers: List[int], max_attempts: int = 50) -> Optional[int]:
        """Find a target that can be solved with given numbers."""
        # Try random targets
        attempted_targets = set()
        
        for _ in range(max_attempts):
            target = random.randint(*self.target_range)
            
            if target in attempted_targets:
                continue
                
            attempted_targets.add(target)
            
            # Check if solvable
            expression, explanation = solve_countdown_puzzle(numbers.copy(), target)
            
            if expression != "No solution found" and not "away from our target" in explanation:
                return target
        
        # If random fails, try systematic search on promising targets
        # (sums/differences of pairs often work well)
        promising_targets = set()
        
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                promising_targets.add(abs(numbers[i] - numbers[j]))
                promising_targets.add(numbers[i] + numbers[j])
        
        # Filter to valid range
        promising_targets = {t for t in promising_targets 
                           if self.target_range[0] <= t <= self.target_range[1] 
                           and t not in attempted_targets}
        
        for target in promising_targets:
            expression, explanation = solve_countdown_puzzle(numbers.copy(), target)
            
            if expression != "No solution found" and not "away from our target" in explanation:
                return target
        
        return None
    
    def format_question(self, numbers: List[int], target: int) -> str:
        """Format the question string."""
        nums_str = ", ".join(map(str, numbers))
        return (f"Using the numbers {nums_str}, create an equation that equals {target}. "
                f"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
                f"Return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.")
    
    def generate_problems(self, num_count: int, count: int = 1000) -> List[dict]:
        """Generate 'count' problems with 'num_count' numbers each."""
        problems = []
        seen_combinations = set()
        
        pbar = tqdm(total=count, desc=f"Generating {num_count}-number problems")
        
        attempts = 0
        max_attempts = count * 100  # Prevent infinite loops
        
        while len(problems) < count and attempts < max_attempts:
            attempts += 1
            
            # Generate unique number combination
            numbers = self.generate_random_numbers(num_count)
            numbers_tuple = tuple(sorted(numbers))
            
            if numbers_tuple in seen_combinations:
                continue
            
            # Find solvable target
            target = self.find_solvable_target(numbers)
            
            if target is None:
                continue
            
            # Generate solution
            expression, _ = solve_countdown_puzzle(numbers.copy(), target)
            
            # Create problem
            problem = {
                'id': f'countdown_custom_{num_count}_{len(problems)+1:03d}',
                'question': self.format_question(numbers, target),
                'solution': expression,
                'answer': str(target)
            }
            
            problems.append(problem)
            seen_combinations.add(numbers_tuple)
            pbar.update(1)
        
        pbar.close()
        
        if len(problems) < count:
            print(f"Warning: Only generated {len(problems)} problems for {num_count} numbers")
        
        return problems

def main():
    """Generate countdown problems for all difficulty levels."""
    output_file = Path("data/countdown_custom.csv")
    metadata_file = Path("data/countdown_custom_metadata.json")
    stats_file = Path("data/countdown_custom_stats.json")
    
    # Ensure data directory exists
    output_file.parent.mkdir(exist_ok=True)
    
    generator = CountdownProblemGenerator()
    all_problems = []
    stats = {}
    
    # Generate problems for each difficulty level
    for num_count in range(4, 11):
        print(f"\n=== Generating problems with {num_count} numbers ===")
        problems = generator.generate_problems(num_count, count=1000)
        all_problems.extend(problems)
        
        stats[f"{num_count}_numbers"] = {
            "count": len(problems),
            "target": 1000,
            "success_rate": len(problems) / 1000
        }
    
    # Save to CSV
    print(f"\nSaving {len(all_problems)} problems to {output_file}")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'question', 'solution', 'answer'])
        writer.writeheader()
        writer.writerows(all_problems)
    
    # Save metadata
    metadata = {
        'source': 'Custom generated countdown problems',
        'total_problems': len(all_problems),
        'difficulty_levels': list(range(4, 11)),
        'number_range': [1, 200],
        'target_range': [1, 100],
        'description': 'Countdown arithmetic puzzles with varying difficulty based on number count',
        'rules': [
            'Use addition (+), subtraction (-), multiplication (*), and division (/)',
            'Each number must be used exactly once',
            'Must reach exactly the target number'
        ]
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save statistics
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nGeneration complete!")
    print(f"- Problems: {output_file}")
    print(f"- Metadata: {metadata_file}")
    print(f"- Statistics: {stats_file}")

if __name__ == "__main__":
    main()