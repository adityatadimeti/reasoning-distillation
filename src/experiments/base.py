from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime
import logging

from src.utils.config import load_config
from src.dashboard.server import DashboardServer

logger = logging.getLogger(__name__)

class BaseExperiment:
    """Base class for all experiments."""
    
    def __init__(
        self, 
        experiment_name: str,
        config: Dict[str, Any],
        dashboard: Optional[DashboardServer] = None
    ):
        """
        Initialize the experiment.
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary
            dashboard: Optional dashboard server for real-time updates
        """
        self.experiment_name = experiment_name
        self.config = config
        self.results = []
        self.dashboard = dashboard
        
        # Create results directory
        self.results_dir = os.path.join(
            self.config.get("results_dir", "results"),
            f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"Results will be saved to {self.results_dir}")
    
    def run(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run the experiment on a list of problems.
        
        Args:
            problems: List of problem dictionaries
            
        Returns:
            List of result dictionaries
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def save_results(self) -> None:
        """Save experiment results to disk."""
        results_path = os.path.join(self.results_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump({
                "experiment_name": self.experiment_name,
                "config": self.config,
                "results": self.results
            }, f, indent=2)
        
        # Save summary metrics
        metrics = self.calculate_metrics()
        metrics_path = os.path.join(self.results_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Results saved to {self.results_dir}")
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate summary metrics for the experiment.
        
        Returns:
            Dictionary of metrics
        """
        # Default implementation: count problems processed
        return {
            "total_problems": len(self.results)
        }