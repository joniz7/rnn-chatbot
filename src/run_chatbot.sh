#!/bin/bash

EXP_PATH=/home/joniktor/experiments/22-4-2016/rnn-chatbot

export CUDA_VISIBLE_DEVICES=0
./src/run_one_chatbot 1 python -u ./src/translate.py --vocab_size=90000 --size=2200 --num_layers=2 --batch_size=64 --max_running_time=40 --embedding_dimensions=300 --checkpoint_dir="../checkpoints/1" --summary_path="../data/summaries/1" > $EXP_PATH/log/log1.txt &
export CUDA_VISIBLE_DEVICES=1
./src/run_one_chatbot 2 python -u ./src/translate.py --vocab_size=90000 --size=2000 --num_layers=3 --batch_size=64 --max_running_time=40 --embedding_dimensions=300 --checkpoint_dir="../checkpoints/2" --summary_path="../data/summaries/2" > $EXP_PATH/log/log2.txt  &
export CUDA_VISIBLE_DEVICES=2
./src/run_one_chatbot 3 python -u ./src/translate.py --vocab_size=90000 --size=1800 --num_layers=4 --batch_size=64 --max_running_time=40 --embedding_dimensions=300 --checkpoint_dir="../checkpoints/3" --summary_path="../data/summaries/3" > $EXP_PATH/log/log3.txt  &
export CUDA_VISIBLE_DEVICES=3
./src/run_one_chatbot 4 python -u ./src/translate.py --vocab_size=90000 --size=1600 --num_layers=5 --batch_size=64 --max_running_time=40 --embedding_dimensions=300 --checkpoint_dir="../checkpoints/4" --summary_path="../data/summaries/4" > $EXP_PATH/log/log4.txt 