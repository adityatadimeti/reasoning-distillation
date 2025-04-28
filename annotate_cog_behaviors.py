#!/usr/bin/env python3
"""
Chunk-based behaviour annotation for long reasoning traces.
-----------------------------------------------------------
* Loads an experiments JSON (format described in prompt).
* Splits each reasoning / summary text into fixed-length sentence chunks with
  optional overlap, annotates each chunk with GPT-4.1, merges the labels,
  and writes an augmented JSON.

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
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

import dotenv
from openai import OpenAI

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


def call_openai(prompt: str, model: str = "gpt-4.1-2025-04-14") -> str:
    """Call OpenAI chat completion with minimal, retry-less wrapper."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def strip_annotations(annotated_text: str) -> str:
    """Remove label markers to compare back to the raw text."""
    return LABEL_RE.sub("", annotated_text).replace("\n", " ").strip()


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text to make comparisons ignore formatting differences."""
    # Replace all whitespace (including newlines) with a single space
    normalized = re.sub(r'\s+', ' ', text)
    return normalized.strip()


def process_text(text: str, chunk_len: int, overlap: int, model: str, flags: Dict[str, Any], verbose: bool = False, ignore_whitespace: bool = True):
    sentences = split_sentences(text)
    chunks = chunk_sentences(sentences, chunk_len, overlap)
    annotated_chunks: List[str] = []
    for ch_idx, chunk in enumerate(chunks):
        raw_chunk = " ".join(chunk)
        prompt = ANNOTATION_PROMPT_TEMPLATE.format(text=raw_chunk)
        annotated = call_openai(prompt, model=model)
        annotated_chunks.append(annotated)

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
            mismatch_info = {
                "chunk_index": ch_idx,
                "expected": raw_chunk,
                "got": cleaned,
            }
            flags.setdefault("mismatch_chunks", []).append(mismatch_info)
            
            if verbose:
                print("\n=== MISMATCH IN CHUNK", ch_idx, "===")
                print("EXPECTED:\n", raw_chunk)
                print("\nGOT:\n", cleaned)
                print("\nDIFF:")
                
                # Print diff based on whitespace handling setting
                if ignore_whitespace:
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
                
    return "\n".join(annotated_chunks)


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
    return p.parse_args()


def main():
    args = parse_args()

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
    
    # Create a new output structure that only includes the selected problems
    output_data = {
        "experiment_name": data.get("experiment_name", ""),
        "problems": []
    }

    for prob in selected:
        print(f"Processing problem {prob['problem_id']} …", flush=True)
        
        # Create a copy of the problem to include in the output
        output_prob = dict(prob)
        output_prob["iterations"] = []
        
        for iteration in prob["iterations"]:
            it_no = iteration["iteration"]
            if iter_set is not None and it_no not in iter_set:
                continue
                
            # Create a copy of the iteration for the output
            output_iter = dict(iteration)
            
            for field in ("reasoning", "post_think_summary"):
                if args.field != "both" and args.field != field:
                    continue
                text = iteration.get(field, "").strip()
                if not text:
                    continue
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
                    print(f"  ⚠️  {num_mismatches} mismatch(es) in {prob['problem_id']} iter {it_no} {field}")
            
            # Only add iterations that were processed
            if iter_set is None or it_no in iter_set:
                output_prob["iterations"].append(output_iter)
        
        # Add the problem with the processed iterations to the output
        output_data["problems"].append(output_prob)

    # write output
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(output_data, fp, indent=2)
    
    print(f"Done. Annotated data written to {out_path}")
    print(f"Included {len(output_data['problems'])} problem(s) from index {args.start} to {end_index}")


if __name__ == "__main__":
    main()