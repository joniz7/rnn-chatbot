#!/bin/bash

echo "now run"
project_run=$(basename "$0")

#create log dir
mkdir log 

submit job

echo "submit"
qsub -cwd \
  -e ./log/$project_run.error \
  -o ./log/$project_run.log \
  ./docker/$project_run

