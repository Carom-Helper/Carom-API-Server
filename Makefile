USR=tglee
PORT_NUM=7576
SRC_NAME=caromapi
APP_NAME=${SRC_NAME}_${USR}
IMAGE_NAME=${SRC_NAME}_image
TARGET_PATH=`pwd`
MODEL_VOLUME = ${TARGET_PATH}:/$(SRC_NAME)
 
# Build and run the container
build:
	@echo "====<PORT_NUM=${PORT_NUM}> <VIDEO_TARGET_PATH=${VIDEO_TARGET_PATH}>==="
	@echo 'docker image build'
	docker image build --build-arg fname=$(SRC_NAME) -t $(IMAGE_NAME) .

run:
	@echo 'docker run -tiu --name="$(APP_NAME)" $(IMAGE_NAME)'
	docker run -ti --name "$(APP_NAME)" --shm-size 32gb --privileged -p $(PORT_NUM):$(PORT_NUM) -v $(MODEL_VOLUME) $(IMAGE_NAME)

stop:
	@echo 'stop docker $(APP_NAME)'
	docker stop $(APP_NAME)
start :
	docker start $(APP_NAME)
exec :
	docker exec -it $(APP_NAME) /bin/bash
runserver :
	docker exec -it $(APP_NAME) python manage.py runserver 0.0.0.0:$(PORT_NUM)
attach:
	docker start $(APP_NAME)
	docker attach $(APP_NAME)
rm:
	@echo 'rm docker $(APP_NAME)'
	docker rm -f $(APP_NAME)

rmi:
	@echo 'rmi docker $(IMAGE_NAME)'
	docker rmi $(IMAGE_NAME)
save:
	docker container commit ${APP_NAME} $(IMAGE_NAME)
	docker save -o ${SRC_NAME}.tar $(IMAGE_NAME)
load:
	docker load -i ${SRC_NAME}.tar
	docker image ls
# rmrmi:
#	docker stop $(APP_NAME) && docker rm $(APP_NAME)
#	docker rmi $(IMAGE_NAME)
