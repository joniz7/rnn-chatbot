#!/bin/bash

docker build -t joniktor/chatbot:gpu -f ./docker/Dockerfile . 

./docker/docker_run_gpu.sh -v /home/joniktor/examples/chatbot1.0/:/root/ --rm -t joniktor/chatbot:gpu