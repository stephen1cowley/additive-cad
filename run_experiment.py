"""
The run_experiment module: uses the config file to run the experiment.
"""

import os
import json
from src.additive_cad import AdditiveCad
from src.experiment_types import ExperimentConfig

if __name__ == "__main__":

    directory = 'experiment_config'
    file_names = os.listdir(directory)
    file_names = [f for f in file_names if os.path.isfile(os.path.join(directory, f))]

    # Take the first file in the /experiment_config directory as the config file
    with open(os.path.join(directory, file_names[0]), "r") as f:
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
