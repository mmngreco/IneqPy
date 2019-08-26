OPT=$1

if [ "$OPT" = "" ];
then
    OPT=-d
fi

docker build -t ineqpy:dev -f ./Dockerfile .
docker run --rm \
    -v `pwd`:/root/git/ineqpy \
    -w /root/git \
    --name test \
    -t $OPT ineqpy:dev \
    /bin/bash

if [ "$OPT" = "-d" ];
then

docker exec test pip install -e ./ineqpy/ && \
docker exec test pip install pysal && \
docker stop test

fi
