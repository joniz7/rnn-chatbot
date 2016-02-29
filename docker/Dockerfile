# image for running tensorflow on titan using the GPU
FROM b.gcr.io/tensorflow/tensorflow:0.7.1-gpu
MAINTAINER Jonathan Orrö <jonathan.orro@gmail.com>

#add source code
ADD ./src /src

WORKDIR /

CMD ./src/run_chatbot.sh