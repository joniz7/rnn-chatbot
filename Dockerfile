FROM b.gcr.io/tensorflow/tensorflow:0.7.0

RUN curl -sL https://deb.nodesource.com/setup_5.x | sudo -E bash -
RUN sudo apt-get install --yes nodejs

ADD package.json package.json
ADD server.js server.js

 ADD src src

RUN npm install

RUN mkdir data; mkdir data/chatlogs; mkdir checkpoints

VOLUME ["data", "checkpoints"]

# EXPOSE 3000

#CMD ["node", "server.js"]