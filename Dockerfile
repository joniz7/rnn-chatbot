FROM kyma/docker-nginx
ADD src/index.html /var/www/html/index.html
CMD 'nginx'