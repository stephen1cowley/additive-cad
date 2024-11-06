"""
The run_experiment module: uses the config file to run the experiment.
"""

import json
from src.additive_cad import AdditiveCad
from src.experiment_types import ExperimentConfig

if __name__ == "__main__":
    with open("experiment_config.json", "r") as f:
        config_data = json.load(f)

    config = ExperimentConfig.from_dict(config_data)
    model = AdditiveCad(config)

    results = model.generate_results()

    with open(f'nq_cad_dola_{str(dola_layers_good)}_{str(dola_layers_bad)}.json', 'w') as json_file:
        json.dump(results, json_file)
        print("Successfully finished the experiment")