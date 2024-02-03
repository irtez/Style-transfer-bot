#!/bin/bash

gunicorn -w ${WORKERS:=1} \
  -b :80 -t ${TIMEOUT:=300} \
  -k uvicorn.workers.UvicornWorker \
  --log-config log.ini \
  main:app