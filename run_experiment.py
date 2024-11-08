"""
The run_experiment module: uses the config file to run the experiment.
"""

import json
import argparse
from src.additive_cad import AdditiveCad
from src.experiment_types import ExperimentConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="../experiment_config/local_testing_conf.json"
    )
    args = parser.parse_args()

    # Take the parsed config file location and load the json
    with open(args.config, "r") as f:
        config_data = json.load(f)

    config = ExperimentConfig.from_dict(config_data)
    model = AdditiveCad(config)

    # Run the experiment
    results = model.generate_results()

    # Write the results to the location specified by the config file
    with open(config.output_file, 'w') as jf:
        json.dump(results, jf)
        print("----------")
        print("Successfully finished the experiment")
