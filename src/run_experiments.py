import yaml
from pathlib import Path
from src.pipeline.main_pipeline import JPXPipeline
from src.utils.logger import setup_logger
from src import CONFIG_DIR

logger = setup_logger('experiments')

def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_experiment(config_path: Path) -> dict:
    """Run an experiment with the specified configuration."""
    # Load configuration
    config = load_config(config_path)
    
    # Create and run pipeline
    pipeline = JPXPipeline(config)
    results = pipeline.run_pipeline()
    
    return results

if __name__ == "__main__":
    # Example configurations for different experiments
    experiments = [
        CONFIG_DIR / 'lightgbm_baseline.yaml',
        CONFIG_DIR / 'lightgbm_optimized.yaml',
        CONFIG_DIR / 'xgboost_baseline.yaml',
        CONFIG_DIR / 'lstm_baseline.yaml'
    ]
    
    # Run all experiments
    for config_path in experiments:
        if config_path.exists():
            logger.info(f"\nRunning experiment with config: {config_path}")
            results = run_experiment(config_path)
            logger.info(f"Results: {results['metrics']}")
        else:
            logger.warning(f"Config file not found: {config_path}")
