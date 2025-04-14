from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime
import logging
from collections import defaultdict

from src.utils.config import load_config
from src.dashboard.server import DashboardServer
from src.llm.base_client import TokenUsage, CostInfo

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
        
        # Initialize token usage and cost tracking
        self.token_usage = {
            "total": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "problems": defaultdict(lambda: defaultdict(int))
        }
        
        self.cost_info = {
            "total": {
                "prompt_cost": 0.0,
                "completion_cost": 0.0,
                "total_cost": 0.0
            },
            "problems": defaultdict(lambda: defaultdict(float))
        }
        
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
    
    def track_token_usage_and_cost(self, problem_id: str, token_usage: TokenUsage, cost_info: CostInfo, 
                             iteration: int = 0, step: str = "reasoning", 
                             continuation_idx: int = None, api_call_idx: int = None) -> None:
        """
        Track token usage and cost for a specific problem and iteration with much more granular detail.
        
        Args:
            problem_id: ID of the problem
            token_usage: TokenUsage object with token counts
            cost_info: CostInfo object with cost information
            iteration: Iteration number (0 = initial, 1 = first improvement, etc.)
            step: Step name (e.g., "reasoning", "summary")
            continuation_idx: Optional index for continuation calls (0, 1, 2, etc.)
            api_call_idx: Optional index for API call within a continuation sequence
        """
        # Update total token usage
        self.token_usage["total"]["prompt_tokens"] += token_usage.prompt_tokens
        self.token_usage["total"]["completion_tokens"] += token_usage.completion_tokens
        self.token_usage["total"]["total_tokens"] += token_usage.total_tokens
        
        # Update problem-specific token usage
        problem_tokens = self.token_usage["problems"][problem_id]
        problem_tokens["prompt_tokens"] += token_usage.prompt_tokens
        problem_tokens["completion_tokens"] += token_usage.completion_tokens
        problem_tokens["total_tokens"] += token_usage.total_tokens
        
        # Update total cost
        self.cost_info["total"]["prompt_cost"] += cost_info.prompt_cost
        self.cost_info["total"]["completion_cost"] += cost_info.completion_cost
        self.cost_info["total"]["total_cost"] += cost_info.total_cost
        
        # Update problem-specific cost
        problem_cost = self.cost_info["problems"][problem_id]
        problem_cost["prompt_cost"] += cost_info.prompt_cost
        problem_cost["completion_cost"] += cost_info.completion_cost
        problem_cost["total_cost"] += cost_info.total_cost
        
        # Store detailed information in the results
        for result in self.results:
            if result.get("problem_id") == problem_id:
                # Initialize token usage and cost tracking for this result if not present
                if "token_usage" not in result:
                    result["token_usage"] = {}
                if "cost_info" not in result:
                    result["cost_info"] = {}
                
                # Create a hierarchical structure for detailed metrics
                # 1. First level: iteration
                iter_key = f"iteration_{iteration}"
                if iter_key not in result["token_usage"]:
                    result["token_usage"][iter_key] = {}
                    result["cost_info"][iter_key] = {}
                
                # 2. Second level: step (reasoning or summary)
                if step not in result["token_usage"][iter_key]:
                    result["token_usage"][iter_key][step] = {}
                    result["cost_info"][iter_key][step] = {}
                
                # 3. Third level: continuations (if applicable)
                if continuation_idx is not None:
                    cont_key = f"continuation_{continuation_idx}"
                    if cont_key not in result["token_usage"][iter_key][step]:
                        result["token_usage"][iter_key][step][cont_key] = {}
                        result["cost_info"][iter_key][step][cont_key] = {}
                    
                    # 4. Fourth level: API calls (if applicable)
                    if api_call_idx is not None:
                        api_key = f"api_call_{api_call_idx}"
                        # Store individual API call details
                        result["token_usage"][iter_key][step][cont_key][api_key] = {
                            "prompt_tokens": token_usage.prompt_tokens,
                            "completion_tokens": token_usage.completion_tokens,
                            "total_tokens": token_usage.total_tokens,
                            "timestamp": datetime.now().isoformat()
                        }
                        result["cost_info"][iter_key][step][cont_key][api_key] = {
                            "prompt_cost": cost_info.prompt_cost,
                            "completion_cost": cost_info.completion_cost,
                            "total_cost": cost_info.total_cost
                        }
                        
                        # Update continuation totals
                        if "total" not in result["token_usage"][iter_key][step][cont_key]:
                            result["token_usage"][iter_key][step][cont_key]["total"] = {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            }
                            result["cost_info"][iter_key][step][cont_key]["total"] = {
                                "prompt_cost": 0.0,
                                "completion_cost": 0.0,
                                "total_cost": 0.0
                            }
                        
                        # Update continuation totals
                        result["token_usage"][iter_key][step][cont_key]["total"]["prompt_tokens"] += token_usage.prompt_tokens
                        result["token_usage"][iter_key][step][cont_key]["total"]["completion_tokens"] += token_usage.completion_tokens
                        result["token_usage"][iter_key][step][cont_key]["total"]["total_tokens"] += token_usage.total_tokens
                        
                        result["cost_info"][iter_key][step][cont_key]["total"]["prompt_cost"] += cost_info.prompt_cost
                        result["cost_info"][iter_key][step][cont_key]["total"]["completion_cost"] += cost_info.completion_cost
                        result["cost_info"][iter_key][step][cont_key]["total"]["total_cost"] += cost_info.total_cost
                    else:
                        # Just store continuation info if no API call index
                        result["token_usage"][iter_key][step][cont_key] = {
                            "prompt_tokens": token_usage.prompt_tokens,
                            "completion_tokens": token_usage.completion_tokens,
                            "total_tokens": token_usage.total_tokens
                        }
                        result["cost_info"][iter_key][step][cont_key] = {
                            "prompt_cost": cost_info.prompt_cost,
                            "completion_cost": cost_info.completion_cost,
                            "total_cost": cost_info.total_cost
                        }
                else:
                    # Just store step info if no continuation
                    result["token_usage"][iter_key][step] = {
                        "prompt_tokens": token_usage.prompt_tokens,
                        "completion_tokens": token_usage.completion_tokens,
                        "total_tokens": token_usage.total_tokens
                    }
                    result["cost_info"][iter_key][step] = {
                        "prompt_cost": cost_info.prompt_cost,
                        "completion_cost": cost_info.completion_cost,
                        "total_cost": cost_info.total_cost
                    }
                
                # Update step totals
                if "total" not in result["token_usage"][iter_key][step]:
                    result["token_usage"][iter_key][step]["total"] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                    result["cost_info"][iter_key][step]["total"] = {
                        "prompt_cost": 0.0,
                        "completion_cost": 0.0,
                        "total_cost": 0.0
                    }
                
                # Update step totals (unless we already counted this in a continuation)
                if continuation_idx is None or "total" not in result["token_usage"][iter_key][step]:
                    result["token_usage"][iter_key][step]["total"]["prompt_tokens"] += token_usage.prompt_tokens
                    result["token_usage"][iter_key][step]["total"]["completion_tokens"] += token_usage.completion_tokens
                    result["token_usage"][iter_key][step]["total"]["total_tokens"] += token_usage.total_tokens
                    
                    result["cost_info"][iter_key][step]["total"]["prompt_cost"] += cost_info.prompt_cost
                    result["cost_info"][iter_key][step]["total"]["completion_cost"] += cost_info.completion_cost
                    result["cost_info"][iter_key][step]["total"]["total_cost"] += cost_info.total_cost
                
                # Update iteration totals
                if "total" not in result["token_usage"][iter_key]:
                    result["token_usage"][iter_key]["total"] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                    result["cost_info"][iter_key]["total"] = {
                        "prompt_cost": 0.0,
                        "completion_cost": 0.0,
                        "total_cost": 0.0
                    }
                
                # Update iteration totals
                result["token_usage"][iter_key]["total"]["prompt_tokens"] += token_usage.prompt_tokens
                result["token_usage"][iter_key]["total"]["completion_tokens"] += token_usage.completion_tokens
                result["token_usage"][iter_key]["total"]["total_tokens"] += token_usage.total_tokens
                
                result["cost_info"][iter_key]["total"]["prompt_cost"] += cost_info.prompt_cost
                result["cost_info"][iter_key]["total"]["completion_cost"] += cost_info.completion_cost
                result["cost_info"][iter_key]["total"]["total_cost"] += cost_info.total_cost
                
                # Also maintain the old format for backward compatibility
                old_iter_key = f"iter{iteration}_{step}"
                result["token_usage"][old_iter_key] = {
                    "prompt_tokens": token_usage.prompt_tokens,
                    "completion_tokens": token_usage.completion_tokens,
                    "total_tokens": token_usage.total_tokens
                }
                result["cost_info"][old_iter_key] = {
                    "prompt_cost": cost_info.prompt_cost,
                    "completion_cost": cost_info.completion_cost,
                    "total_cost": cost_info.total_cost
                }
                
                # Update overall problem totals
                if "total" not in result["token_usage"]:
                    result["token_usage"]["total"] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                if "total" not in result["cost_info"]:
                    result["cost_info"]["total"] = {
                        "prompt_cost": 0.0,
                        "completion_cost": 0.0,
                        "total_cost": 0.0
                    }
                
                result["token_usage"]["total"]["prompt_tokens"] += token_usage.prompt_tokens
                result["token_usage"]["total"]["completion_tokens"] += token_usage.completion_tokens
                result["token_usage"]["total"]["total_tokens"] += token_usage.total_tokens
                
                result["cost_info"]["total"]["prompt_cost"] += cost_info.prompt_cost
                result["cost_info"]["total"]["completion_cost"] += cost_info.completion_cost
                result["cost_info"]["total"]["total_cost"] += cost_info.total_cost
                
                break
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate summary metrics for the experiment.
        
        Returns:
            Dictionary of metrics
        """
        # Default implementation: count problems processed and include token/cost metrics
        metrics = {
            "total_problems": len(self.results),
            "token_usage": self.token_usage["total"],
            "cost_info": self.cost_info["total"],
            "problems": {}
        }
        
        # Include detailed iteration-level metrics for each problem
        for result in self.results:
            problem_id = result.get("problem_id")
            if not problem_id:
                continue  # Skip results without a problem_id
                
            # Initialize problem entry with totals from the experiment-level tracking
            # Use empty dictionaries as fallbacks if the problem isn't in the tracking
            metrics["problems"][problem_id] = {
                "token_usage": {
                    "total": dict(self.token_usage["problems"].get(problem_id, {}))
                },
                "cost_info": {
                    "total": dict(self.cost_info["problems"].get(problem_id, {}))
                }
            }
            
            # Add iteration-level metrics if available
            if "token_usage" in result and isinstance(result["token_usage"], dict):
                for key, usage in result["token_usage"].items():
                    if key != "total" and usage:  # Skip the total and empty values
                        metrics["problems"][problem_id]["token_usage"][key] = usage
            
            if "cost_info" in result and isinstance(result["cost_info"], dict):
                for key, cost in result["cost_info"].items():
                    if key != "total" and cost:  # Skip the total and empty values
                        metrics["problems"][problem_id]["cost_info"][key] = cost
        
        return metrics