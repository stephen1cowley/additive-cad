import json
from src.additive_cad import ExperimentConfig

with open("experiment_config.json", "r") as f:
    config_data = json.load(f)

# Create Config instance from dictionary
config = ExperimentConfig.from_dict(config_data)

print(config)

