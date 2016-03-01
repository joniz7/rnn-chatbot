# image for running tensorflow on titan using the GPU
FROM b.gcr.io/tensorflow/tensorflow:0.7.0-gpu
MAINTAINER Jonathan Orrö <jonathan.orro@gmail.com>

#add source code
ADD ./src /src

#dont know how this works, so outcommented
#VOLUME ["/data"]
#VOLUME ["/checkpoints"]

WORKDIR /

CMD ./src/run_chatbot.sh