#!/bin/sh

aws ecr-public --profile forecapp get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws

docker build -t forecapp-api .

aws ecr --profile forecapp get-login-password --region us-west-1 | docker login --username AWS --password-stdin 719834076613.dkr.ecr.us-west-1.amazonaws.com

docker tag forecapp-api:latest 719834076613.dkr.ecr.us-west-1.amazonaws.com/forecapp-api:latest
docker push 719834076613.dkr.ecr.us-west-1.amazonaws.com/forecapp-api:latest
