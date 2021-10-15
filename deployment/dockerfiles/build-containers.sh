#!/bin/bash

BUGLAB_ROOT_PATH=$1
echo "BugLab Root Path is $BUGLAB_ROOT_PATH"

docker build -t buglabcr.azurecr.io/buglab/buglab-base -f $BUGLAB_ROOT_PATH/deployment/dockerfiles/baseimage.Dockerfile $BUGLAB_ROOT_PATH
