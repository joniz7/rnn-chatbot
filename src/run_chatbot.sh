#!/bin/bash
echo "source!!"
python -u ./src/translate.py --vocab_size=30000 --size=1024 --num_layers=3 --batch_size=64 --max_running_time=40 --embedding_dimensions=300