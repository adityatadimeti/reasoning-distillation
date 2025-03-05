"""
Prompt templates for summarizing reasoning traces.
"""
from typing import Dict, Tuple, Callable
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """Prompt template for summarization."""
    system_template: str
    user_template: str
    
    def format(self, **kwargs) -> Tuple[str, str]:
        """
        Format the prompt templates with the provided parameters.
        
        Args:
            **kwargs: Key-value parameters for formatting
            
        Returns:
            Tuple of (formatted system prompt, formatted user prompt)
        """
        system_prompt = self.system_template.format(**kwargs)
        user_prompt = self.user_template.format(**kwargs)
        return system_prompt, user_prompt

# Default prompt for summarization
DEFAULT_PROMPT = PromptTemplate(
    system_template="""You are an expert at summarizing mathematical reasoning traces.
Your goal is to produce concise but complete summaries of reasoning traces that:
1. Preserve all key reasoning steps and intermediate calculations
2. Highlight potential errors or invalid assumptions in the original reasoning
3. Identify what paths of reasoning were explored and abandoned
4. Maintain the logical flow and dependencies between steps
5. Avoid introducing new reasoning or calculations not present in the original trace
6. Format the summary in a clear, step-by-step manner
7. Include key equations and numerical results with proper notation

Your summary should be detailed enough that another model could understand the full reasoning process,
but concise enough to eliminate redundancy and verbose explanations.""",
    
    user_template="""Below is a reasoning trace from a math problem. 
Summarize this reasoning trace while preserving all key steps, intermediate calculations, 
and potential errors in the original reasoning.

REASONING TRACE:
{reasoning_trace}

SUMMARY:"""
)

# Concise prompt for very brief summaries
CONCISE_PROMPT = PromptTemplate(
    system_template="""You are an expert at creating extremely concise summaries of mathematical reasoning.
Your goal is to produce the most concise summary possible while still preserving the essential logical flow.
Focus only on the most critical steps and results, removing all unnecessary explanation and elaboration.
Include only:
1. The core problem formulation
2. The key mathematical steps taken (equations and transformations)
3. The final result or conclusion

Your summary should be no more than 1/4 the length of the original reasoning trace.""",
    
    user_template="""Provide an extremely concise summary of this mathematical reasoning trace.
Focus only on the most critical steps and results.

REASONING TRACE:
{reasoning_trace}

CONCISE SUMMARY:"""
)

# Error-focused prompt for identifying reasoning mistakes
ERROR_FOCUSED_PROMPT = PromptTemplate(
    system_template="""You are an expert at analyzing mathematical reasoning for errors and flaws.
Your goal is to summarize a reasoning trace with special emphasis on identifying:
1. Calculation errors
2. Invalid mathematical operations or transformations
3. Logical fallacies or incorrect deductions
4. Misunderstandings of the original problem
5. Unjustified assumptions
6. Overlooked edge cases or constraints

First identify any errors, then summarize the reasoning trace accurately,
highlighting where errors occurred and how they affected the solution.
If the reasoning appears correct, explicitly state this in your summary.""",
    
    user_template="""Below is a mathematical reasoning trace. 
Create a summary that focuses on identifying and explaining any errors or flaws in the reasoning.
If the reasoning appears correct, explicitly state this in your summary.

REASONING TRACE:
{reasoning_trace}

ERROR-FOCUSED SUMMARY:"""
)

# Key steps prompt for extracting the core reasoning path
KEY_STEPS_PROMPT = PromptTemplate(
    system_template="""You are an expert at extracting the key steps from mathematical reasoning.
Your goal is to identify and clearly present the critical steps in the reasoning process while
removing all redundant explanation and exploration.
Focus on:
1. The initial understanding and setup of the problem
2. Major transformations or insights that advance the solution
3. The core equations and calculations that lead to the result
4. The final answer and its justification

Present these key steps in a numbered list format for clarity.""",
    
    user_template="""Extract and present the key steps from this mathematical reasoning trace.
Use a numbered list format for clarity.

REASONING TRACE:
{reasoning_trace}

KEY STEPS:"""
)

# Dictionary mapping prompt names to their templates
PROMPTS = {
    "default": DEFAULT_PROMPT,
    "concise": CONCISE_PROMPT,
    "error_focused": ERROR_FOCUSED_PROMPT,
    "key_steps": KEY_STEPS_PROMPT
}

def get_summarization_prompt(prompt_name: str) -> PromptTemplate:
    """
    Get a summarization prompt template by name.
    
    Args:
        prompt_name: Name of the prompt template
        
    Returns:
        PromptTemplate instance
        
    Raises:
        ValueError: If prompt name is not recognized
    """
    if prompt_name not in PROMPTS:
        raise ValueError(f"Unknown summarization prompt: {prompt_name}")
    
    return PROMPTS[prompt_name]