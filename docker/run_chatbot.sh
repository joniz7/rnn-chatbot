#!/bin/bash

#docker build -t joniktor/chatbot:gpu -f ./docker/Dockerfile . 
echo "docker"
./docker/docker_run_gpu.sh -v /home/joniktor/examples/chatbot1.0/data:/root/data --rm -t joniktor/chatbot