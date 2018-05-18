#!/bin/bash
python3 VAE_inference.py $1 $2 0 models/VAE1_model.pkt
python3 GAN_inference.py $1 $2 0 models/G_model.pkt
python3 ACGAN_inference.py $1 $2 0 models/ACG2_model.pkt