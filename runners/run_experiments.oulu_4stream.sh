#!/usr/bin/env bash

CONFIG_DIR=../oulu/config
RESULTS_DIR=../oulu/results/4stream
PREDICTIONS_DIR=$RESULTS_DIR/predictions
BEST_MODEL_DIR=$RESULTS_DIR/best_models
PLOTS_DIR=$RESULTS_DIR/plots

# create necessary directories
mkdir -p $CONFIG_DIR
mkdir -p $BEST_MODEL_DIR
mkdir -p $PLOTS_DIR
mkdir -p $PREDICTIONS_DIR

START=1
END=10

if [ $# -ne 1 ]; then
echo "USAGE: ./run_experiments.oulu_4stream.sh ./experiments/oulu_4stream_experiments.txt"
exit
fi

EXPERIMENT_FILE=$1
for line in `cat $EXPERIMENT_FILE`; do
FIRST_CHAR=${line:0:1}
if [ $FIRST_CHAR != "#" ]; then
RUNNER=`echo $line | cut -d ',' -f 1`
EXPERIMENT_NAME=`echo $line | cut -d ',' -f 2`
echo "runner=$RUNNER experiment=$EXPERIMENT_NAME"
for i in $(eval echo "{$START..$END}"); do python $RUNNER --config $CONFIG_DIR/$EXPERIMENT_NAME.ini --current_runtime $i --write_results $RESULTS_DIR/$EXPERIMENT_NAME.$i.txt --save_predictions $PREDICTIONS_DIR/$EXPERIMENT_NAME.$i.txt --save_best $BEST_MODEL_DIR/$EXPERIMENT_NAME.$i.pkl --save_plot $PLOTS_DIR/$EXPERIMENT_NAME.$i; done
fi;
done
