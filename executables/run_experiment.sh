#!/bin/bash
source ~/pytorch-env/bin/activate
export TRANSFORMERS_CACHE=/rds/user/ssc42/hpc-work

cd ..
python3 run_experiment.py --config $1
