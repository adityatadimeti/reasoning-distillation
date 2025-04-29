#!/usr/bin/env python3
"""
Chunk-based behaviour annotation for long reasoning traces.
-----------------------------------------------------------
* Loads an experiments JSON (format described in prompt).
* Splits each reasoning / summary text into fixed-length sentence chunks with
  optional overlap, annotates each chunk with GPT-4.1, merges the labels,
  and writes an augmented JSON.
* Saves progress incrementally to avoid data loss on interruption.

Usage (examples)
----------------
python chunk_annotate.py --json data/aime_dataset.json \
       --start 0 --end 59 --iters all \
       --field reasoning --chunk 40 --overlap 5

Environment
-----------
OPENAI_API_KEY is loaded from a .env file in the working directory (python-dotenv).
"""

import argparse
import json
import os
import re
import sys
import datetime
import time
import concurrent.futures
import random
import tempfile
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import dotenv
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
ANNOTATION_PROMPT_TEMPLATE = (
    """Please split the following reasoning chain of an LLM into annotated parts using labels and the following format [\"label\"]...[\"end-section\"]. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels.\n"""
    "Available labels:\n"
    "0. initializing -> The model is rephrasing the given task and states initial thoughts.\n"
    "1. deduction -> The model is performing a deduction step based on its current approach and assumptions.\n"
    "2. adding-knowledge -> The model is enriching the current approach with recalled facts.\n"
    "3. example-testing -> The model generates examples to test its current approach.\n"
    "4. uncertainty-estimation -> The model is stating its own uncertainty.\n"
    "5. backtracking -> The model decides to change its approach.\n"
    "The reasoning chain to analyze:\n{text}\n\n"
    "Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out."
)

LABEL_RE = re.compile(r"\[\"(?:\d+.\s*)(?:initializing|deduction|adding-knowledge|example-testing|uncertainty-estimation|backtracking)\"\]|\[\"end-section\"\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Retry configuration
MAX_RETRIES = 500
RETRY_BASE_DELAY = 1  # seconds
MAX_WORKERS = 16 # maximum number of parallel API calls

def split_sentences(text: str) -> List[str]:
    """Very lightweight sentence splitter (good enough for LLM‑generated prose)."""
    parts: List[str] = SENTENCE_SPLIT_RE.split(text.strip())
    # keep punctuation that was consumed by the regex
    return [p.strip() for p in parts if p.strip()]


def chunk_sentences(sentences: List[str], length: int, overlap: int) -> List[List[str]]:
    """Return list of sentence chunks with `overlap` sentences between consecutive chunks."""
    if length <= 0:
        raise ValueError("chunk length must be > 0")
    if overlap >= length:
        raise ValueError("overlap must be smaller than chunk length")

    chunks: List[List[str]] = []
    start = 0
    n = len(sentences)
    while start < n:
        end = min(start + length, n)
        chunks.append(sentences[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def call_openai_with_retry(prompt: str, model: str = "gpt-4.1-2025-04-14") -> str:
    """Call OpenAI chat completion with retry logic for transient errors."""
    client = OpenAI()
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        
        except (APIError, RateLimitError, APIConnectionError) as e:
            # Don't retry on last attempt
            if attempt == MAX_RETRIES - 1:
                print(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
                raise
            
            # Calculate exponential backoff with jitter
            delay = min(RETRY_BASE_DELAY * (2 ** attempt) + (0.1 * random.random()), 60)
            print(f"API error: {str(e)}. Retrying in {delay:.2f} seconds (attempt {attempt+1}/{MAX_RETRIES})...")
            time.sleep(delay)
        
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise
            
    # Should never reach here due to the raise in the loop
    raise RuntimeError("Failed to get a response after multiple retries")


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text to make comparisons ignore formatting differences."""
    # Replace all whitespace (including newlines) with a single space
    normalized = re.sub(r'\s+', ' ', text)
    return normalized.strip()


def is_labeled_word_diff(original: str, annotated: str) -> bool:
    """Check if the difference is just a word being prefixed with a label."""
    # Extract the word part after the label
    label_match = re.search(r'\["[^"]+"\](.+)', annotated)
    if label_match:
        word_after_label = label_match.group(1).strip()
        # Check if the original text starts with this word
        return original.strip().startswith(word_after_label)
    return False


def process_chunk(chunk_idx: int, chunk: List[str], model: str, ignore_whitespace: bool) -> Tuple[int, str, Dict[str, Any]]:
    """Process a single chunk and return the annotated text with any flags."""
    raw_chunk = " ".join(chunk)
    prompt = ANNOTATION_PROMPT_TEMPLATE.format(text=raw_chunk)
    
    flags: Dict[str, Any] = {}
    annotated = call_openai_with_retry(prompt, model=model)
    
    # simple quality check for this chunk
    cleaned = strip_annotations(annotated)
    
    # Check for mismatch based on ignore_whitespace setting
    if ignore_whitespace:
        normalized_raw = normalize_whitespace(raw_chunk)
        normalized_cleaned = normalize_whitespace(cleaned)
        is_mismatch = normalized_cleaned != normalized_raw
    else:
        is_mismatch = cleaned.strip() != raw_chunk.strip()
        
    if is_mismatch:
        # Check if the difference is just due to the expected labeling pattern
        # This needs a more complex diff analysis than just comparing the full strings
        import difflib
        normalized_raw = normalize_whitespace(raw_chunk)
        normalized_cleaned = normalize_whitespace(cleaned)
        
        # Get word-level differences
        expected_words = re.findall(r'\S+', normalized_raw)
        got_words = re.findall(r'\S+', normalized_cleaned)
        word_diffs = list(difflib.ndiff(expected_words, got_words))
        
        # Check if all differences are just label additions or end-section markers
        real_differences = False
        i = 0
        while i < len(word_diffs):
            diff = word_diffs[i]
            
            # Skip if this is just an end-section marker being added
            if diff[0] == '+' and '[end-section]' in diff:
                i += 1
                continue
                
            # Check for a removed word followed by a labeled version of the same word
            if diff[0] == '-' and i+1 < len(word_diffs) and word_diffs[i+1][0] == '+':
                removed_word = diff[2:]
                added_word = word_diffs[i+1][2:]
                
                # Case 1: Word has been labeled - ["label"]word
                if re.match(r'\["[^"]+"\].+', added_word) and \
                   added_word.strip('"[]0123456789. ').endswith(removed_word):
                    i += 2  # Skip both diffs
                    continue
                
                # Case 2: Word followed by [end-section] - word[end-section]
                if removed_word in added_word and '[end-section]' in added_word:
                    i += 2  # Skip both diffs
                    continue
                
                real_differences = True
                break
            elif diff[0] != ' ':  # Any other difference
                real_differences = True
                break
            else:
                i += 1
        
        if real_differences:
            mismatch_info = {
                "chunk_index": chunk_idx,
                "expected": raw_chunk,
                "got": cleaned,
            }
            flags["mismatch_chunks"] = [mismatch_info]
    
    return chunk_idx, annotated, flags


def strip_annotations(annotated_text: str) -> str:
    """Remove label markers to compare back to the raw text."""
    return LABEL_RE.sub("", annotated_text).replace("\n", " ").strip()


def process_text(text: str, chunk_len: int, overlap: int, model: str, flags: Dict[str, Any], verbose: bool = False, ignore_whitespace: bool = True):
    sentences = split_sentences(text)
    chunks = chunk_sentences(sentences, chunk_len, overlap)
    
    # Process chunks in parallel
    annotated_chunks = [""] * len(chunks)
    chunk_flags = {}
    
    print(f"  Processing {len(chunks)} chunks in parallel...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, idx, chunk, model, ignore_whitespace): idx
            for idx, chunk in enumerate(chunks)
        }
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                chunk_idx, annotated, chunk_flag = future.result()
                annotated_chunks[chunk_idx] = annotated
                
                # Merge flags
                if chunk_flag.get("mismatch_chunks"):
                    flags.setdefault("mismatch_chunks", []).extend(chunk_flag["mismatch_chunks"])
                    
                    if verbose and chunk_flag.get("mismatch_chunks"):
                        for mismatch in chunk_flag["mismatch_chunks"]:
                            ch_idx = mismatch["chunk_index"]
                            raw_chunk = mismatch["expected"]
                            cleaned = mismatch["got"]
                            
                            print("\n=== MISMATCH IN CHUNK", ch_idx, "===")
                            print("EXPECTED:\n", raw_chunk)
                            print("\nGOT:\n", cleaned)
                            print("\nDIFF:")
                            
                            # Print diff based on whitespace handling setting
                            if ignore_whitespace:
                                normalized_raw = normalize_whitespace(raw_chunk)
                                normalized_cleaned = normalize_whitespace(cleaned)
                                print("Normalized comparison (ignoring whitespace):")
                                print(f"EXPECTED: {normalized_raw[:100]}{'...' if len(normalized_raw) > 100 else ''}")
                                print(f"GOT: {normalized_cleaned[:100]}{'...' if len(normalized_cleaned) > 100 else ''}")
                                
                                # Show content differences with words
                                import difflib
                                expected_words = re.findall(r'\S+', normalized_raw)
                                got_words = re.findall(r'\S+', normalized_cleaned)
                                diff = list(difflib.ndiff(expected_words, got_words))
                                
                                print("\nWord differences:")
                                has_diff = False
                                for d in diff:
                                    if d[0] != ' ':
                                        has_diff = True
                                        print(d)
                                if not has_diff:
                                    print("No actual content differences detected, only whitespace variations.")
                            else:
                                from difflib import ndiff
                                for i, s in enumerate(ndiff(raw_chunk.splitlines(), cleaned.splitlines())):
                                    if s[0] == ' ':  # unchanged
                                        continue
                                    elif s[0] == '-':
                                        print(f"- {s[2:]}")
                                    elif s[0] == '+':
                                        print(f"+ {s[2:]}")
                            print("=" * 40)
                
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                # Continue with other chunks even if one fails
    
    return "\n".join(annotated_chunks)


def save_progress(output_data: Dict[str, Any], out_path: str, temp_prefix: str = "_temp_") -> None:
    """
    Save the current progress safely using atomic write pattern:
    1. Write to a temporary file first
    2. Rename the temporary file to the target file
    This ensures we don't corrupt our output file if the script is interrupted during writing.
    """
    # Create temporary file in the same directory
    temp_path = f"{out_path}{temp_prefix}{os.getpid()}"
    
    try:
        # Write to temporary file
        with open(temp_path, "w", encoding="utf-8") as fp:
            json.dump(output_data, fp, indent=2)
        
        # Rename temp file to target file (atomic operation)
        shutil.move(temp_path, out_path)
        
    except Exception as e:
        # Clean up temp file if something goes wrong
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        print(f"Error saving progress: {str(e)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Chunk-annotate reasoning traces using GPT-4.1.")
    p.add_argument("--json", required=True, help="Path to the input experiment JSON file")
    p.add_argument("--start", type=int, default=0, help="First problem index (inclusive)")
    p.add_argument("--end", type=int, default=None, help="Last problem index (inclusive). Default = last problem")
    p.add_argument("--iters", default="all", help="Iterations to process: 'all' or e.g. 0,1,2")
    p.add_argument("--field", choices=["reasoning", "post_think_summary", "both"], default="reasoning")
    p.add_argument("--chunk", type=int, default=40, help="Chunk length in sentences")
    p.add_argument("--overlap", type=int, default=5, help="Sentence overlap between chunks")
    p.add_argument("--model", default="gpt-4.1-2025-04-14", help="OpenAI model name (default gpt-4.1-2025-04-14)")
    p.add_argument("--out", help="Output JSON path (default: <input>_annotated.json)")
    p.add_argument("--verbose", "-v", action="store_true", help="Print detailed information about mismatches")
    p.add_argument("--strict", action="store_true", help="Use strict comparison (don't ignore whitespace)")
    p.add_argument("--max-workers", type=int, default=MAX_WORKERS, 
                  help=f"Maximum number of parallel API calls (default: {MAX_WORKERS})")
    p.add_argument("--max-retries", type=int, default=MAX_RETRIES,
                  help=f"Maximum number of API call retries (default: {MAX_RETRIES})")
    p.add_argument("--resume", action="store_true", help="Resume from existing output file if it exists")
    return p.parse_args()


def load_partial_results(out_path: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Load and parse existing partial results from the output file.
    Returns a tuple of (data, processed_problem_ids)
    """
    if not os.path.exists(out_path):
        return None, []
    
    try:
        with open(out_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            
        # Extract the IDs of problems that have already been processed
        processed_ids = [prob["problem_id"] for prob in data.get("problems", [])]
        return data, processed_ids
    
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error loading existing output file: {str(e)}")
        print("Will start from scratch.")
        return None, []


def main():
    args = parse_args()
    
    # Update global configuration based on command line args
    global MAX_WORKERS, MAX_RETRIES
    MAX_WORKERS = args.max_workers
    MAX_RETRIES = args.max_retries

    # load env
    dotenv.load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in environment or .env", file=sys.stderr)
        sys.exit(1)

    with open(args.json, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    problems: List[Dict[str, Any]] = data["problems"]
    end_index = args.end if args.end is not None else len(problems) - 1
    selected = problems[args.start : end_index + 1]

    if args.iters == "all":
        iter_set = None
    else:
        iter_set = {int(x) for x in args.iters.split(",")}

    out_path = args.out or Path(args.json).with_name(Path(args.json).stem + "_annotated.json")
    
    # Whether to ignore whitespace in mismatch detection
    ignore_whitespace = not args.strict
    
    # Check if we should resume from an existing output file
    existing_data = None
    processed_problem_ids = []
    
    if args.resume:
        existing_data, processed_problem_ids = load_partial_results(out_path)
        if existing_data:
            print(f"Resuming from existing output file with {len(processed_problem_ids)} problems already processed.")
            output_data = existing_data
        else:
            print("No valid existing output file found. Starting from scratch.")
            args.resume = False
    
    # Create a new output structure if not resuming
    if not args.resume or not existing_data:
        output_data = {
            "experiment_name": data.get("experiment_name", ""),
            "annotation_parameters": {
                "input_file": args.json,
                "start_index": args.start,
                "end_index": end_index,
                "iterations": args.iters,
                "field": args.field,
                "chunk_size": args.chunk,
                "chunk_overlap": args.overlap,
                "model": args.model,
                "strict_comparison": args.strict,
                "max_workers": MAX_WORKERS,
                "max_retries": MAX_RETRIES,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "problems": []
        }

    for prob_idx, prob in enumerate(selected):
        prob_id = prob['problem_id']
        
        # Skip already processed problems when resuming
        if args.resume and prob_id in processed_problem_ids:
            print(f"Skipping already processed problem {prob_id} ({prob_idx+1}/{len(selected)})")
            continue
            
        print(f"Processing problem {prob_id} ({prob_idx+1}/{len(selected)}) …", flush=True)
        
        # Create a copy of the problem to include in the output
        output_prob = dict(prob)
        output_prob["iterations"] = []
        
        # Flag to track if any iterations were processed for this problem
        problem_processed = False
        
        for iteration in prob["iterations"]:
            it_no = iteration["iteration"]
            if iter_set is not None and it_no not in iter_set:
                continue
                
            # Create a copy of the iteration for the output
            output_iter = dict(iteration)
            iteration_processed = False
            
            for field in ("reasoning", "post_think_summary"):
                if args.field != "both" and args.field != field:
                    continue
                text = iteration.get(field, "").strip()
                if not text:
                    continue
                    
                print(f"  Processing {prob_id} iteration {it_no} {field}...")
                
                flags: Dict[str, Any] = {}
                annotated = process_text(
                    text,
                    chunk_len=args.chunk,
                    overlap=args.overlap,
                    model=args.model,
                    flags=flags,
                    verbose=args.verbose,
                    ignore_whitespace=ignore_whitespace
                )
                output_iter[f"{field}_annotated"] = annotated
                output_iter.setdefault("annotation_flags", {})[field] = flags
                if flags.get("mismatch_chunks"):
                    num_mismatches = len(flags["mismatch_chunks"])
                    print(f"  ⚠️  {num_mismatches} mismatch(es) in {prob_id} iter {it_no} {field}")
                
                iteration_processed = True
            
            # Only add iterations that were processed
            if iteration_processed and (iter_set is None or it_no in iter_set):
                output_prob["iterations"].append(output_iter)
                problem_processed = True
                
                # Save progress after each iteration
                if problem_processed:
                    # Add current problem to output data
                    temp_output_data = dict(output_data)
                    
                    # Find if this problem already exists in the output and replace it
                    problem_exists = False
                    for i, p in enumerate(temp_output_data.get("problems", [])):
                        if p["problem_id"] == prob_id:
                            temp_output_data["problems"][i] = output_prob
                            problem_exists = True
                            break
                    
                    # Add problem if it doesn't exist
                    if not problem_exists:
                        temp_output_data["problems"].append(output_prob)
                    
                    # Update last update timestamp
                    temp_output_data["annotation_parameters"]["last_update"] = datetime.datetime.now().isoformat()
                    
                    # Save to disk
                    save_progress(temp_output_data, out_path)
                    print(f"  Progress saved after iteration {it_no}")
        
        # After processing all iterations for a problem
        if problem_processed:
            # Only add the problem to output_data if not saving per iteration
            # (otherwise it's already been added)
            problem_exists = False
            for i, p in enumerate(output_data.get("problems", [])):
                if p["problem_id"] == prob_id:
                    output_data["problems"][i] = output_prob
                    problem_exists = True
                    break
            
            if not problem_exists:
                output_data["problems"].append(output_prob)
            
            # Update timestamp and save progress after each problem
            output_data["annotation_parameters"]["last_update"] = datetime.datetime.now().isoformat()
            save_progress(output_data, out_path)
            print(f"Problem {prob_id} completed and progress saved.")

    # Final save with completion timestamp
    output_data["annotation_parameters"]["completion_timestamp"] = datetime.datetime.now().isoformat()
    save_progress(output_data, out_path)
    
    print(f"Done. Annotated data written to {out_path}")
    print(f"Included {len(output_data['problems'])} problem(s) from index {args.start} to {end_index}")


if __name__ == "__main__":
    main()