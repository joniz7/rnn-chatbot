#!/bin/bash

docker build -t joniktor/chatbot:gpu -f ./docker/Dockerfile . 

./docker/docker_run_gpu.sh --rm -t joniktor/chatbot:gpu