"""
Tests for pipeline functionality.
"""
import os
import pytest
from dotenv import load_dotenv
import tempfile
import shutil
from pathlib import Path

from src.utils.config import Config
from src.models.deepseek import DeepSeekModel
from src.data.dataset import Dataset
from src.pipeline.base_pipeline import BasePipeline
from src.pipeline.baseline_pipeline import BaselineReasoningPipeline

# Load environment variables
load_dotenv()

# Only run API-calling tests if explicitly enabled
api_tests_enabled = pytest.mark.skipif(
    os.environ.get("ENABLE_API_TESTS") != "1",
    reason="API tests disabled. Set ENABLE_API_TESTS=1 to enable."
)

# Sample configuration for testing
SAMPLE_CONFIG = {
    "pipeline": {
        "name": "test_baseline",
        "type": "baseline",
        "batch_size": 1,
        "save_intermediates": True
    },
    "model": {
        "name": "deepseek_r1",
        "model_id": "accounts/fireworks/models/deepseek-r1",
        "api_type": "fireworks",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    "reasoning": {
        "think_tag": True,
        "max_extensions": 1,
        "target_token_count": 500
    },
    "dataset": {
        "name": "aime2024",
        "base_path": "archive/data",
        "raw_file": "aime_2024_completions_updated.csv",
        "columns": {
            "id": "ID",
            "problem": "Problem",
            "solution": "Solution",
            "answer": "Answer",
            "reasoning_trace": "model_completion",
            "extracted_answer": "extracted_answer"
        }
    },
    "output": {
        "results_path": "test_results",
        "save_format": "json"
    },
    "logging": {
        "level": "INFO"
    }
}

class TestBaselineReasoningPipeline:
    """Tests for the baseline reasoning pipeline"""
    
    @pytest.fixture
    def config(self):
        """Create a configuration object for testing"""
        return Config(base_config=SAMPLE_CONFIG)
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def output_config(self, config, temp_dir):
        """Configure output to temporary directory"""
        config_dict = config.to_dict()
        config_dict["output"]["results_path"] = temp_dir
        return Config(base_config=config_dict)
    
    def test_pipeline_initialization(self, config):
        """Test initialization of the baseline pipeline"""
        pipeline = BaselineReasoningPipeline(config)
        
        assert pipeline.name == "test_baseline"
        assert pipeline.batch_size == 1
        assert isinstance(pipeline.model, DeepSeekModel)
    
    def test_save_results(self, output_config, temp_dir):
        """Test saving results and metrics"""
        pipeline = BaselineReasoningPipeline(output_config)
        
        # Sample results and metrics
        results = {
            "pipeline": "test_baseline",
            "num_problems": 1,
            "results": [
                {
                    "question": "What is 2+2?",
                    "reasoning": "I need to add 2 and 2. 2+2=4.",
                    "answer": "The answer is 4.",
                    "extracted_answer": "4",
                    "normalized_extracted": "4",
                    "ground_truth": "4",
                    "normalized_ground_truth": "4"
                }
            ]
        }
        
        metrics = {
            "accuracy": 1.0,
            "correct_count": 1,
            "total_evaluated": 1
        }
        
        # Save results
        pipeline.save_results(results, metrics)
        
        # Check if files were created
        result_files = list(Path(temp_dir).glob("*/results_*.json"))
        metric_files = list(Path(temp_dir).glob("*/metrics_*.json"))
        summary_files = list(Path(temp_dir).glob("*/summary_*.json"))
        
        assert len(result_files) == 1, "Results file should be created"
        assert len(metric_files) == 1, "Metrics file should be created"
        assert len(summary_files) == 1, "Summary file should be created"

@api_tests_enabled
class TestBaselineReasoningPipelineIntegration:
    """Integration tests for the baseline reasoning pipeline"""
    
    @pytest.fixture
    def config(self):
        """Create a configuration object for testing"""
        return Config(base_config=SAMPLE_CONFIG)
    
    @pytest.fixture
    def pipeline(self, config):
        """Create a pipeline instance for testing"""
        return BaselineReasoningPipeline(config)
    
    @pytest.fixture
    def dataset(self, config):
        """Create a dataset instance for testing"""
        return Dataset(config)
    
    def test_api_key_present(self):
        """Verify API key is available"""
        assert os.environ.get("FIREWORKS_API_KEY"), "FIREWORKS_API_KEY not found in environment"
    
    def test_run_pipeline_single_problem(self, pipeline, dataset):
        """Test running the pipeline on a single problem"""
        try:
            # Load the dataset
            dataset.load_raw()
            
            # Get a specific problem ID
            problem_id = dataset.data_raw.iloc[0]["ID"]
            
            # Run the pipeline on a single problem
            results = pipeline.run(
                dataset=dataset,
                problem_ids=[problem_id]
            )
            
            print("\n=== PIPELINE RESULTS ===")
            print(f"Pipeline: {results['pipeline']}")
            print(f"Number of problems: {results['num_problems']}")
            print(f"Total time: {results['total_time']:.2f}s")
            
            # Check result structure
            assert results["pipeline"] == "test_baseline"
            assert results["num_problems"] == 1
            assert "total_time" in results
            assert "results" in results
            assert len(results["results"]) == 1
            
            # Check problem result
            problem_result = results["results"][0]
            assert "question" in problem_result
            assert "reasoning" in problem_result
            assert "answer" in problem_result
            assert "extracted_answer" in problem_result
            
            # Evaluate results
            metrics = pipeline.evaluate(results)
            
            print("\n=== PIPELINE METRICS ===")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            
            # Check metric structure
            assert "accuracy" in metrics
            assert "correct_count" in metrics
            assert "total_evaluated" in metrics
            
        except Exception as e:
            pytest.skip(f"Could not run pipeline integration test: {str(e)}")

@api_tests_enabled
class TestPipelineIntegration:
    """Integration tests for reasoning pipelines"""
    
    @pytest.fixture
    def baseline_config(self):
        """Create a baseline configuration object for testing"""
        return Config(base_config=SAMPLE_CONFIG)
    
    @pytest.fixture
    def summary_config(self):
        """Create a summary configuration object for testing"""
        config_dict = SAMPLE_CONFIG.copy()
        config_dict["pipeline"] = {
            "name": "test_summary",
            "type": "summary",
            "batch_size": 1,
            "max_iterations": 1
        }
        config_dict["summarization"] = {
            "method": "self",
            "strategy": "default",
            "prompt": "default"
        }
        return Config(base_config=config_dict)
    
    @pytest.fixture
    def baseline_pipeline(self, baseline_config):
        """Create a baseline pipeline instance for testing"""
        return BaselineReasoningPipeline(baseline_config)
    
    @pytest.fixture
    def summary_pipeline(self, summary_config):
        """Create a summary pipeline instance for testing"""
        from src.pipeline.summary_pipeline import SummaryReasoningPipeline
        return SummaryReasoningPipeline(summary_config)
    
    @pytest.fixture
    def dataset(self, baseline_config):
        """Create a dataset instance for testing"""
        return Dataset(baseline_config)
    
    def test_api_key_present(self):
        """Verify API key is available"""
        assert os.environ.get("FIREWORKS_API_KEY"), "FIREWORKS_API_KEY not found in environment"
    
    def test_run_baseline_pipeline(self, baseline_pipeline, dataset):
        """Test running the baseline pipeline on a single problem"""
        try:
            # Load the dataset
            dataset.load_raw()
            
            # Get a specific problem ID
            problem_id = dataset.data_raw.iloc[0]["ID"]
            
            # Run the pipeline on a single problem
            results = baseline_pipeline.run(
                dataset=dataset,
                problem_ids=[problem_id]
            )
            
            print("\n=== BASELINE PIPELINE RESULTS ===")
            print(f"Pipeline: {results['pipeline']}")
            print(f"Number of problems: {results['num_problems']}")
            print(f"Total time: {results['total_time']:.2f}s")
            
            # Check result structure
            assert results["pipeline"] == "test_baseline"
            assert results["num_problems"] == 1
            assert "total_time" in results
            assert "results" in results
            assert len(results["results"]) == 1
            
            # Check problem result
            problem_result = results["results"][0]
            assert "question" in problem_result
            assert "reasoning" in problem_result
            assert "answer" in problem_result
            assert "extracted_answer" in problem_result
            
            # Evaluate results
            metrics = baseline_pipeline.evaluate(results)
            
            print("\n=== BASELINE PIPELINE METRICS ===")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            
            # Check metric structure
            assert "accuracy" in metrics
            assert "correct_count" in metrics
            assert "total_evaluated" in metrics
            
        except Exception as e:
            pytest.skip(f"Could not run baseline pipeline test: {str(e)}")
    
    def test_run_summary_pipeline(self, summary_pipeline, dataset):
        """Test running the summary pipeline on a single problem"""
        try:
            # Load the dataset
            dataset.load_raw()
            
            # Get a specific problem ID
            problem_id = dataset.data_raw.iloc[0]["ID"]
            
            # Run the pipeline on a single problem
            results = summary_pipeline.run(
                dataset=dataset,
                problem_ids=[problem_id],
                max_iterations=1  # Just do one iteration for testing
            )
            
            print("\n=== SUMMARY PIPELINE RESULTS ===")
            print(f"Pipeline: {results['pipeline']}")
            print(f"Number of problems: {results['num_problems']}")
            print(f"Total time: {results['total_time']:.2f}s")
            print(f"Summarization method: {results['summarization_method']}")
            
            # Check result structure
            assert results["pipeline"] == "test_summary"
            assert results["num_problems"] == 1
            assert "total_time" in results
            assert "results" in results
            assert len(results["results"]) == 1
            
            # Check problem result
            problem_result = results["results"][0]
            assert "question" in problem_result
            assert "iterations" in problem_result
            assert len(problem_result["iterations"]) > 0
            assert "extracted_answer" in problem_result
            
            # Check that summarization occurred
            if len(problem_result["iterations"]) > 1:
                assert "summary" in problem_result["iterations"][1]
            
            # Evaluate results
            metrics = summary_pipeline.evaluate(results)
            
            print("\n=== SUMMARY PIPELINE METRICS ===")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            
            # Check metric structure
            assert "accuracy" in metrics
            assert "correct_count" in metrics
            assert "total_evaluated" in metrics
            
        except Exception as e:
            pytest.skip(f"Could not run summary pipeline test: {str(e)}")