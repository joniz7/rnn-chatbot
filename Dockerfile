FROM kyma/docker-nginx
ADD src/index.html /var/www/index.html
CMD 'nginx'