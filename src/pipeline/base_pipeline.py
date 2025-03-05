"""
Base pipeline class for reasoning enhancement experiments.
"""
from abc import ABC, abstractmethod
import time
import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from src.utils.config import Config, PROJECT_ROOT
from src.models.base import Model
from src.data.dataset import Dataset

logger = logging.getLogger(__name__)

class BasePipeline(ABC):
    """
    Abstract base class for all experiment pipelines.
    """
    def __init__(self, config: Config):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.name = config.get("pipeline.name", "unknown_pipeline")
        
        # Set up logging
        log_level = getattr(logging, config.get("logging.level", "INFO"))
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set up output directory
        output_config = config.get("output", {})
        self.results_path = Path(output_config.get("results_path", "results"))
        if not self.results_path.is_absolute():
            self.results_path = PROJECT_ROOT / self.results_path
        
        self.pipeline_results_path = self.results_path / self.name
        
        # Create output directory if it doesn't exist
        os.makedirs(self.pipeline_results_path, exist_ok=True)
        
        # Other configuration
        self.save_intermediates = config.get("pipeline.save_intermediates", True)
    
    @abstractmethod
    def run(self, dataset: Dataset, **kwargs) -> Dict[str, Any]:
        """
        Run the pipeline on the provided dataset.
        
        Args:
            dataset: Dataset instance
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with pipeline results
        """
        pass
    
    @abstractmethod
    def evaluate(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the results of the pipeline.
        
        Args:
            results: Pipeline results
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    def save_results(self, results: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """
        Save pipeline results and metrics to disk.
        
        Args:
            results: Pipeline results
            metrics: Evaluation metrics
        """
        timestamp = int(time.time())
        
        # Save results
        results_file = self.pipeline_results_path / f"results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save metrics
        metrics_file = self.pipeline_results_path / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save a summary file
        summary = {
            "timestamp": timestamp,
            "pipeline": self.name,
            "config": self.config.to_dict(),
            "metrics": metrics
        }
        
        summary_file = self.pipeline_results_path / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")
        logger.info(f"Saved metrics to {metrics_file}")
        logger.info(f"Saved summary to {summary_file}")
    
    def log_results(self, metrics: Dict[str, Any]) -> None:
        """
        Log metrics and results to console.
        
        Args:
            metrics: Evaluation metrics
        """
        logger.info("=== Pipeline Results ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")