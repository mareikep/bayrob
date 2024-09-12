.PHONY: help check build run compose run-img
.DEFAULT: help

# name of docker container and image
NAME = bayrob-web

help:
	@echo ""
	@echo "Use 'make <target>' where <target> is one of:"
	@echo ""
	@echo "  build	    build the docker container \`$(SRC)-img\`"
	@echo "  run	    run the docker container \`$(SRC)-container\`"
	@echo "  compose	build the docker container \`$(SRC)-container\` from image \`$(SRC)-img\`"
	@echo "  clean		remove all docker containers, networks, unused volumes, dangling images and unused build cache"
	@echo ""
	@echo "Go forth and make something great!"
	@echo ""

# check if docker is installed and print a hint if its not
check:
	@docker --version >> /dev/null || ( echo "\nDocker is not installed!\n"; exit 1 )

build: check
	@docker build --tag $(NAME)-img .

run-img: check
	@docker run --rm -ti -p 5005:5005 $(NAME)-img

run: check build compose

compose: check
	@docker compose up -d

clean: check
	@docker system prune --volumes
