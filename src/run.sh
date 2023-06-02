#!/bin/bash
GPU=0
LOG_FILE=./experiments${GPU}.out
DATASETS=('../data/*')
MODELS=('lstm' 'tcn' 'mlp')
MODELS_ML=('rf' 'xgb' 'lr')
PARAMETERS=./parameters_reduced.json
OUTPUT=../results
CSV_FILENAME=results.csv

python generate_data.py
python main_dl.py --datasets ${DATASETS[@]} --models ${MODELS[@]} --models_ml ${MODELS_ML[@]} --gpu ${GPU} --parameters  $PARAMETERS --output $OUTPUT --csv_filename $CSV_FILENAME > $LOG_FILE 2>&1 &
