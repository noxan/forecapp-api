#!/bin/sh

ssh -i "aws-forecapp.pem" ubuntu@ec2-54-219-210-155.us-west-1.compute.amazonaws.com << EOF

cd forecapp-api/

git pull

.venv/bin/pip install -r requirements.txt

sudo systemctl restart gunicorn.socket

EOF
