FROM kyma/docker-nginx
#ADD src/index.html /var/www/html/index.html
#ADD nginx.conf /etc/nginx/nginx.conf

RUN mkdir /var/www
ADD src/* /var/www/

CMD 'nginx'