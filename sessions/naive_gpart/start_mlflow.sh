#!/bin/bash

mkdir -p ./mlruns
mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlruns --host 0.0.0.0 --port 6200 

