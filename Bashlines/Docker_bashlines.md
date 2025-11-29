                                    DOCKER BASHLINES


# Create new image
docker build -t prediction_model .


_____________________________

Deletes unused content 

# Remove stopped containers
docker container prune -f

# Remove dangling images (unused)
docker image prune -a -f

# Remove unused volumes
docker volume prune -f

# Remove unused networks
docker network prune -f

____________________________

Deletes everything built 

# Or everything not in use at all:

docker system prune -a --volumes -f
