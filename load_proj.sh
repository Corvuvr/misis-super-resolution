#!/bin/bash

# Get datasets
mkdir datasets 
curl -sSL https://www.kaggle.com/api/v1/datasets/download/jesucristo/super-resolution-benchmarks | bsdtar -xf- -C datasets
mv datasets/Set14/Set14/original/ datasets/Set14/Set14/Set14

# Get models
git clone https://github.com/Corvuvr/Real-ESRGAN
git clone https://github.com/Corvuvr/RT4KSR