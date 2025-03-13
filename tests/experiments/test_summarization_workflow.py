import pytest
import json
from unittest.mock import MagicMock, patch
from src.experiments.summarization import SummarizationExperiment
from src.reasoning.extractor import extract_reasoning_trace

# Patch the model clients at the class level to prevent real API calls
@pytest.fixture(autouse=True)
def mock_fireworks_client():
    """Mock FireworksModelClient to prevent real API calls."""
    with patch('src.llm.fireworks_client.FireworksModelClient') as mock:
        # Configure the mock to return itself when instantiated
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_config():
    """Fixture to provide a mock configuration."""
    return {
        "experiment_name": "test_experiment",
        "results_dir": "./test_results",
        "reasoning_model": "deepseek-r1",  # Use a recognized model name
        "summarizer_model": "deepseek-v3", # Use a recognized model name
        "summarizer_type": "external",     # Set to external to use the summarizer_model
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": 40,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "reasoning_prompt_template": "Solve this problem: {question}",
        "summarize_prompt_template": "Summarize this reasoning: {reasoning}",
        "improved_prompt_template": "Solve this problem again: {question}\n\nPrevious attempts: {summaries}",
        "max_iterations": 2  # Set to allow multiple iterations
    }

@pytest.fixture
def mock_problem():
    """Fixture to provide a mock problem."""
    return {
        "id": "test_problem_1",
        "question": "What is 2+2?",
        "answer": "4"
    }

@pytest.fixture
def mock_think_output():
    """Fixture to provide mock model output with think tags."""
    return """
    <think>
    Let me solve this step by step.
    2 + 2 = 4
    </think>
    
    The answer is \\boxed{4}.
    """

@pytest.fixture
def mock_think_output_wrong():
    """Fixture to provide mock model output with an incorrect answer."""
    return """
    <think>
    Let me solve this step by step.
    2 + 2 = 5
    Hmm, that doesn't look right.
    </think>
    
    The answer is \\boxed{5}.
    """

@pytest.fixture
def mock_summary():
    """Fixture to provide a mock summary."""
    return "The model incorrectly calculated 2+2=5."

@patch('src.llm.model_factory.create_model_client')
def test_accumulate_summaries(mock_create_client, mock_fireworks_client, mock_config, mock_problem, 
                              mock_think_output, mock_think_output_wrong, mock_summary):
    """Test that summaries are accumulated across iterations correctly."""
    
    # Create mock model instances
    mock_reasoning_model = MagicMock()
    mock_summarizer_model = MagicMock()
    
    # Set up create_model_client to return our mock models
    mock_create_client.side_effect = [mock_reasoning_model, mock_summarizer_model]
    
    # Configure mock responses
    # Iteration 0: Generate incorrect answer
    mock_reasoning_model.generate_response.side_effect = [
        mock_think_output_wrong,  # First attempt (wrong answer)
        mock_think_output         # Second attempt (correct answer)
    ]
    
    # Set up summarizer to return our mock summary
    mock_summarizer_model.generate_response.return_value = mock_summary
    
    # Mock extract_reasoning_trace to return properly extracted content
    with patch('src.reasoning.extractor.extract_reasoning_trace') as mock_extract:
        # First call: extract from wrong output
        # Second call: extract from correct output
        mock_extract.side_effect = [
            "Let me solve this step by step.\n2 + 2 = 5\nHmm, that doesn't look right.",
            "Let me solve this step by step.\n2 + 2 = 4"
        ]
        
        # Create the experiment
        experiment = SummarizationExperiment(
            experiment_name="test_summarization",
            config=mock_config
        )
        
        # Process the problem
        result = experiment._process_problem(mock_problem)
    
    # Verify the result structure
    assert result["problem_id"] == "test_problem_1"
    assert len(result["iterations"]) == 2  # Should have two iterations
    
    # Check iteration 0
    assert result["iterations"][0]["iteration"] == 0
    assert result["iterations"][0]["reasoning"] == mock_think_output_wrong
    assert result["iterations"][0]["answer"] == "5"
    assert result["iterations"][0]["correct"] is False
    
    # Check iteration 1 
    assert result["iterations"][1]["iteration"] == 1
    assert result["iterations"][1]["reasoning"] == mock_think_output
    assert result["iterations"][1]["answer"] == "4"
    assert result["iterations"][1]["correct"] is True
    
    # Most importantly, check that the improved prompt contains the summary
    # Extract the calls made to generate_response
    calls = mock_reasoning_model.generate_response.call_args_list
    
    # The second call should be for iteration 1 with the improved prompt
    improved_prompt = calls[1][0][0]  # First arg of second call
    
    # Verify that the improved prompt contains the summary from iteration 0
    assert "ATTEMPT 0 SUMMARY" in improved_prompt
    assert mock_summary in improved_prompt

@patch('src.llm.model_factory.create_model_client')
def test_reasoning_trace_extraction(mock_create_client, mock_fireworks_client, mock_config, mock_problem):
    """Test that reasoning traces are correctly extracted from model outputs."""
    
    # Create mock model instances
    mock_reasoning_model = MagicMock()
    mock_summarizer_model = MagicMock()
    
    # Set up the mock model factory to return our mock models
    mock_create_client.side_effect = [mock_reasoning_model, mock_summarizer_model]
    
    # Configure mock responses with multiple think tags
    multi_think_output = """
    <think>
    First attempt: 2 + 2 = 5
    </think>
    
    Wait, I made a mistake. Let me recalculate.
    
    <think>
    Second attempt: 2 + 2 = 4
    </think>
    
    The answer is \\boxed{4}.
    """
    
    # Configure model responses
    mock_reasoning_model.generate_response.return_value = multi_think_output
    mock_summarizer_model.generate_response.return_value = "Summary of both attempts."
    
    # Create and configure the experiment for no iterations (just to test extraction)
    mock_config["max_iterations"] = 0  # No additional iterations
    experiment = SummarizationExperiment(
        experiment_name="test_extraction",
        config=mock_config
    )
    
    # Mock the extraction function
    extracted_trace = "First attempt: 2 + 2 = 5\n\nSecond attempt: 2 + 2 = 4"
    with patch('src.reasoning.extractor.extract_reasoning_trace') as mock_extract:
        mock_extract.return_value = extracted_trace
        
        # Process a sample problem with patched summarize_reasoning
        with patch('src.reasoning.summarizer.summarize_reasoning') as mock_summarize:
            mock_summarize.return_value = "Summary with extracted content"
            
            # Process the problem
            experiment._process_problem(mock_problem)
            
            # Verify extract_reasoning_trace was called with allow_fallback=False
            mock_extract.assert_called_with(multi_think_output, allow_fallback=False)
            
            # Check if summarize_reasoning was called with the extracted content
            mock_summarize.assert_called()
            # The reasoning parameter should be the extracted trace
            assert mock_summarize.call_args[0][1] == extracted_trace

@patch('src.llm.model_factory.create_model_client')
def test_extraction_failure_handling(mock_create_client, mock_fireworks_client, mock_config, mock_problem):
    """Test that the experiment properly handles extraction failures."""
    
    # Create mock model instances
    mock_reasoning_model = MagicMock()
    mock_summarizer_model = MagicMock()
    
    # Set up the mock model factory to return our mock models
    mock_create_client.side_effect = [mock_reasoning_model, mock_summarizer_model]
    
    # Configure reasoning model response with no think tags
    no_think_output = "This is a response without any think tags. The answer is \\boxed{5}."
    mock_reasoning_model.generate_response.return_value = no_think_output
    
    # Create the experiment
    experiment = SummarizationExperiment(
        experiment_name="test_extraction_failure",
        config=mock_config
    )
    
    # Mock the extraction function to return None (simulating failure)
    with patch('src.reasoning.extractor.extract_reasoning_trace') as mock_extract:
        mock_extract.return_value = None
        
        # Process should raise ValueError when extraction fails
        with pytest.raises(ValueError, match="Could not extract reasoning trace for problem"):
            # Ensure we disable dashboard to hit the non-dashboard code path
            experiment.dashboard = None
            experiment._process_problem(mock_problem) 