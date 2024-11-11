"""
The run_experiment module: uses the json config file (see src/experiment_types for the schema) to
run the experiment.
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
        default="../experiment_config/local_testing_conf.json",
        help="""
        Path to the json config file. For the schema see /src/experiment_types For examples see 
        /experiment_config/.
        """
    )
    args = parser.parse_args()

    # Take the parsed config file location and load the json
    with open(args.config, "r") as f:
        config_data = json.load(f)

    config = ExperimentConfig.from_dict(config_data)  # unmarshall
    model = AdditiveCad(config)

    # Run the experiment
    results = model.generate_results()

    # Write the results to the location specified by the config file
    with open(config.output_file, 'w') as jf:
        json.dump(results, jf)
        print("----------")
        print("Successfully finished the experiment")
