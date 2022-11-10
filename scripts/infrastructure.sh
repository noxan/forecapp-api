#/bin/sh

aws ecr --profile forecapp create-repository --repository-name forecapp-api --image-scanning-configuration scanOnPush=true --image-tag-mutability MUTABLE
